"""FastAPI sub-routers, split out of main.py so a feature owns one file.

``main.py`` includes each router with a single ``app.include_router(...)`` line —
keeping the monolith's edit surface small when several features land at once.
"""
