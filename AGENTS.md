# AGENTS.md

## Purpose
Guide AI agents contributing to EmpathyChat.

### Coding Rules
- Follow TypeScript + Python dual stack conventions.
- Keep all UI components in `packages/ui`.
- Shared state in `packages/state` with Zustand.
- LLM interactions are mocked via `packages/api`.

### Testing
- Jest for TypeScript logic
- Pytest for Python server

### Safe Extensions
When adding empathy blending logic:
- Ensure test coverage
- Avoid storing user data or secrets
