# Location: backend/tests/test_auth.py
import os
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

os.environ["SECRET_KEY"]   = "a" * 32
os.environ["GROQ_API_KEY"] = "test-key-not-real"
os.environ["DATABASE_URL"] = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://myuser:mypassword@localhost:5432/opsiq"
)
os.environ["DB_SCHEMA"] = "opsiq_test"

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text
from sqlalchemy.pool import NullPool
from auth.models import Base, User
from auth.database import get_db
from config import DATABASE_URL, DB_SCHEMA
from fastapi import FastAPI
from auth.router import router as auth_router, limiter

# Minimal test app — no lifespan, no background tasks
auth_app = FastAPI()
auth_app.include_router(auth_router)
auth_app.state.limiter = limiter
limiter.enabled = False

# Separate engine for tests — NullPool avoids stale asyncpg connections across tests
_engine_kwargs: dict = {"echo": False, "poolclass": NullPool}
if DATABASE_URL.startswith("postgresql") and DB_SCHEMA:
    _engine_kwargs["connect_args"] = {"server_settings": {"search_path": DB_SCHEMA}}

test_engine = create_async_engine(DATABASE_URL, **_engine_kwargs)
TestSession = async_sessionmaker(test_engine, expire_on_commit=False, class_=AsyncSession)


async def override_get_db():
    async with TestSession() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


auth_app.dependency_overrides[get_db] = override_get_db


# Create schema + tables ONCE for the whole test session
@pytest_asyncio.fixture(scope="session", autouse=True)
async def create_schema():
    async with test_engine.begin() as conn:
        await conn.execute(text("CREATE SCHEMA IF NOT EXISTS opsiq_test"))
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA IF EXISTS opsiq_test CASCADE"))
    await test_engine.dispose()


# Truncate users table before each test so tests don't interfere
@pytest_asyncio.fixture(autouse=True)
async def clean_tables():
    async with test_engine.begin() as conn:
        await conn.execute(text("TRUNCATE opsiq_test.users RESTART IDENTITY CASCADE"))
    yield


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=auth_app),
        base_url="http://test",
    ) as c:
        yield c


# ── Signup ────────────────────────────────────────────────────────────────────

async def test_signup_success(client):
    res = await client.post("/auth/signup", json={
        "username": "topu", "email": "topu@test.com", "password": "password123"
    })
    assert res.status_code == 201
    data = res.json()
    assert data["email"] == "topu@test.com"
    assert data["username"] == "topu"
    assert "id" in data


async def test_signup_sets_cookie(client):
    res = await client.post("/auth/signup", json={
        "username": "topu", "email": "topu@test.com", "password": "password123"
    })
    assert res.status_code == 201
    assert "opsiq_session" in res.cookies


async def test_signup_duplicate_email_returns_409(client):
    payload = {"username": "topu", "email": "topu@test.com", "password": "password123"}
    await client.post("/auth/signup", json=payload)
    res = await client.post("/auth/signup", json=payload)
    assert res.status_code == 409
    assert "email" in res.json()["detail"].lower()


async def test_signup_same_username_different_email_allowed(client):
    await client.post("/auth/signup", json={
        "username": "topu", "email": "topu1@test.com", "password": "password123"
    })
    res = await client.post("/auth/signup", json={
        "username": "topu", "email": "topu2@test.com", "password": "password123"
    })
    assert res.status_code == 201


async def test_signup_short_password_returns_422(client):
    res = await client.post("/auth/signup", json={
        "username": "topu", "email": "topu@test.com", "password": "short"
    })
    assert res.status_code == 422


async def test_signup_short_username_returns_422(client):
    res = await client.post("/auth/signup", json={
        "username": "ab", "email": "topu@test.com", "password": "password123"
    })
    assert res.status_code == 422


async def test_signup_invalid_email_returns_422(client):
    res = await client.post("/auth/signup", json={
        "username": "topu", "email": "notanemail", "password": "password123"
    })
    assert res.status_code == 422


# ── Login ─────────────────────────────────────────────────────────────────────

async def test_login_success(client):
    await client.post("/auth/signup", json={
        "username": "topu", "email": "topu@test.com", "password": "password123"
    })
    res = await client.post("/auth/login", json={
        "email": "topu@test.com", "password": "password123"
    })
    assert res.status_code == 200
    assert res.json()["email"] == "topu@test.com"


async def test_login_sets_cookie(client):
    await client.post("/auth/signup", json={
        "username": "topu", "email": "topu@test.com", "password": "password123"
    })
    res = await client.post("/auth/login", json={
        "email": "topu@test.com", "password": "password123"
    })
    assert "opsiq_session" in res.cookies


async def test_login_wrong_password_returns_401(client):
    await client.post("/auth/signup", json={
        "username": "topu", "email": "topu@test.com", "password": "password123"
    })
    res = await client.post("/auth/login", json={
        "email": "topu@test.com", "password": "wrongpassword"
    })
    assert res.status_code == 401


async def test_login_unknown_email_returns_401(client):
    res = await client.post("/auth/login", json={
        "email": "nobody@test.com", "password": "password123"
    })
    assert res.status_code == 401


async def test_login_error_message_does_not_leak_which_field_is_wrong(client):
    await client.post("/auth/signup", json={
        "username": "topu", "email": "topu@test.com", "password": "password123"
    })
    res1 = await client.post("/auth/login", json={
        "email": "nobody@test.com", "password": "password123"
    })
    res2 = await client.post("/auth/login", json={
        "email": "topu@test.com", "password": "wrongpassword"
    })
    assert res1.json()["detail"] == res2.json()["detail"]


# ── Me ────────────────────────────────────────────────────────────────────────

async def test_me_returns_user_info(client):
    await client.post("/auth/signup", json={
        "username": "topu", "email": "topu@test.com", "password": "password123"
    })
    res = await client.get("/auth/me")
    assert res.status_code == 200
    data = res.json()
    assert data["email"] == "topu@test.com"
    assert "created_at" in data


async def test_me_unauthenticated_returns_401(client):
    res = await client.get("/auth/me")
    assert res.status_code == 401


# ── Logout ────────────────────────────────────────────────────────────────────

async def test_logout_clears_cookie(client):
    await client.post("/auth/signup", json={
        "username": "topu", "email": "topu@test.com", "password": "password123"
    })
    await client.post("/auth/logout")
    res = await client.get("/auth/me")
    assert res.status_code == 401