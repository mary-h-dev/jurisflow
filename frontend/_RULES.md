# frontend/_RULES.md

## Stack

- Next.js 14 (App Router)
- TypeScript (strict mode)
- Tailwind CSS
- shadcn/ui components
- React Query (TanStack Query) for server state
- Zustand for client state

---

## Project Layout

```
frontend/
├── app/
│   ├── (auth)/
│   │   ├── login/
│   │   └── register/
│   ├── (onboarding)/
│   │   ├── role/
│   │   └── legal-fields/
│   ├── (dashboard)/
│   │   ├── search/          # legal search chat
│   │   ├── analyze/         # case analysis with agents
│   │   └── documents/       # upload & manage documents
│   └── layout.tsx
├── components/
│   ├── ui/                  # shadcn/ui base components
│   ├── auth/                # login, register forms
│   ├── search/              # chat interface, source cards
│   ├── agents/              # debate viewer, case report
│   └── documents/           # upload zone, status tracker
├── lib/
│   ├── api.ts               # typed API client (fetch wrapper)
│   ├── auth.ts              # JWT storage + refresh logic
│   └── types.ts             # shared TypeScript types
├── hooks/
│   ├── useAuth.ts
│   ├── useSearch.ts
│   ├── useAgents.ts
│   └── useDocuments.ts
└── stores/
    └── auth.store.ts        # Zustand auth store
```

---

## Routing Rules

- Use Next.js 14 **App Router** exclusively — no Pages Router
- Route groups with parentheses for layout separation: `(auth)`, `(dashboard)`
- Protected routes check JWT in `middleware.ts`
- Redirect unauthenticated users to `/login`
- Redirect incomplete onboarding to `/onboarding/role`

---

## Component Rules

- All components must be **TypeScript** with explicit prop types
- Use `'use client'` only when necessary (interactivity, hooks)
- Server Components by default for data fetching
- Never fetch data inside client components — use React Query hooks
- Persian text must use `font-vazir` or equivalent Persian font

---

## API Client Rules

- All backend calls go through `lib/api.ts`
- JWT token stored in `httpOnly` cookie (not localStorage)
- Auto-refresh token on 401 response
- Always handle loading and error states

```typescript
// lib/api.ts pattern
export async function searchQuery(query: string, law: string) {
  const res = await apiClient.post('/search/query', { query, law })
  return res.data as SearchOut
}
```

---

## UI/UX Rules

- RTL layout for all pages (`dir="rtl"`)
- Persian numerals where appropriate
- Confidence level badges:
  - `high` → green
  - `medium` → yellow
  - `low` → red + lawyer recommendation
- Always show citation sources below AI answers
- Agent debate must be collapsible (not shown by default)

---

## Key Pages

### `/search`
- Chat-style interface
- Show answer + confidence badge + source articles
- Law selector dropdown (قانون مدنی, ...)

### `/analyze`
- Text input or document selector
- Progress indicator for each agent node
- Collapsible debate section (defender / prosecutor / judge)
- Final recommendation card with next steps

### `/documents`
- Drag & drop upload zone (PDF + images)
- Real-time status polling every 3 seconds
- "Analyze" button appears when status = completed

### `/onboarding`
- Step 1: role selection (lawyer / client)
- Step 2: legal field multi-select with icons
- Visual node graph showing selections (frontend only)

---

## Forbidden

- No `localStorage` for JWT tokens
- No inline styles — use Tailwind classes only
- No `any` TypeScript type
- No direct `fetch()` calls outside `lib/api.ts`
- No LTR layout — this is a Persian product
