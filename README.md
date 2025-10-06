# MindShift 

MindShift is an app that helps users see situations from multiple perspectives using AI-driven persona roles.

## 🧠 Concept
Users describe a situation and choose roles (e.g., “Husband” / “Wife”). The system prepends these roles to the input prompt and generates responses that help understand each side.

## 🚀 Quickstart

```bash
npm install
npm run dev:web     # start web
npm run dev:mobile  # start Expo mobile
```

## 🧩 Architecture Overview
- **Expo (React Native + Web)** for shared frontend
- **FastAPI** backend for mock LLM API
- **Zustand** for state
- **Tailwind + shadcn/ui** for UI
- **Testing:** Jest (TypeScript), Pytest (Python)

## 🧪 Testing
```bash
npm test
pytest
```

## 📦 Deployment
Web build via Expo web export. Mobile via Expo Go or EAS build.

---

© 2025 MindShift MVP
