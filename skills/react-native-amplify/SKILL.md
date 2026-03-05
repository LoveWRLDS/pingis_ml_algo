---
name: react-native-amplify
description: "Set up AWS Amplify Gen 1 (v6) in a React Native CLI app with Cognito auth, AppSync GraphQL API, and S3 storage. IMPORTANT: This skill covers Amplify Gen 1 / v6 only — NOT Gen 2 (which uses a completely different file-based config). Use when creating or adding AWS backend to a React Native app, setting up user authentication with email/password, creating a GraphQL data model with @model/@auth directives, or integrating S3 file uploads. Covers the full flow from `amplify init` to working CRUD operations."
---

# React Native + AWS Amplify Setup

Complete guide to integrate AWS Amplify Gen 1 (v6) into a React Native CLI project, covering Auth (Cognito), API (AppSync/GraphQL), and Storage (S3).

## Prerequisites

- AWS account with CLI access
- `npm install -g @aws-amplify/cli` (v12+)
- `amplify configure` already run (IAM user created)
- React Native CLI project (not Expo)

## Step 1: Initialize Amplify

```bash
cd YourProject
amplify init
```

Prompts:
- Name: your project name
- Environment: `staging` (or `dev`)
- Default editor: pick yours
- App type: `javascript`
- Framework: `react-native`
- Source directory: `src`
- Distribution directory: `build`
- Build command: `npm run build`
- Start command: `npm start`

## Step 2: Add Auth (Cognito)

```bash
amplify add auth
```

Recommended config:
- Default configuration → **Email** as sign-in
- Required attributes: `email`, `name` (add `profile` if storing role)
- MFA: OFF (or SMS for production)
- Password min length: 6+

## Step 3: Add API (AppSync GraphQL)

```bash
amplify add api
```

Choose:
- **GraphQL**
- Auth: **API Key** (default) + **Amazon Cognito User Pool** (additional)
- Schema: edit now

### Schema Design Pattern

See [references/schema-patterns.md](references/schema-patterns.md) for the full schema pattern reference including `@auth` rules and relation directives.

Key conventions:
- Every model gets `@model` + `@auth` directives
- Owner rule: `{ allow: owner }` — only creator can CRUD
- Private read: `{ allow: private, operations: [read] }` — any authenticated user can read
- Multi-owner read: `{ allow: owner, ownerField: "viewers", operations: [read] }` — shared access via `viewers: [String]`
- Relations: `@hasMany` / `@belongsTo`

### Example schema template (copy to `amplify/backend/api/<name>/schema.graphql`):

```graphql
type User @model @auth(rules: [
  { allow: owner },
  { allow: private, operations: [read] }
]) {
  id: ID!
  email: String
  name: String
  avatarKey: String
  # Add your relations:
  # items: [Item] @hasMany
}

type Item @model @auth(rules: [
  { allow: owner },
  { allow: owner, ownerField: "viewers", operations: [read] }
]) {
  id: ID!
  name: String!
  description: String
  imageKey: String
  user: User @belongsTo
  viewers: [String]
}
```

## Step 4: Add Storage (S3)

```bash
amplify add storage
```

- Content (images, audio, video)
- Access: Auth users only (read/write)
- Guest: read-only (if needed for public assets)

## Step 5: Push to AWS

```bash
amplify push
```

This creates:
- `src/aws-exports.js` — auto-generated config
- `src/graphql/` — auto-generated queries, mutations, subscriptions

## Step 6: Install Dependencies

```bash
npm install aws-amplify @aws-amplify/react-native @react-native-async-storage/async-storage react-native-get-random-values
```

For iOS:
```bash
cd ios && pod install && cd ..
```

## Step 7: Configure in App

In `index.js` (before `AppRegistry`):

```javascript
import { Amplify } from 'aws-amplify';
import amplifyconfig from './src/aws-exports';

Amplify.configure(amplifyconfig);
```

## Step 8: Implement Auth

See [references/auth-patterns.md](references/auth-patterns.md) for the complete auth implementation guide.

Key imports:
```typescript
import { signIn, signUp, signOut, getCurrentUser, fetchUserAttributes } from 'aws-amplify/auth';
```

Auth flow:
1. `signUp({ username: email, password, options: { userAttributes: { email, name } } })`
2. User gets email verification code → `confirmSignUp({ username, confirmationCode })`
3. `signIn({ username: email, password })` → returns `{ isSignedIn }`
4. `getCurrentUser()` → check if already logged in
5. `fetchUserAttributes()` → get profile data (email, name, profile/role)
6. `signOut()` → clear session

## Step 9: Implement GraphQL CRUD

See [references/graphql-patterns.md](references/graphql-patterns.md) for the full CRUD pattern reference.

Key pattern — always use `generateClient` with explicit `userPool` auth:

```typescript
import { generateClient } from 'aws-amplify/api';

const client = generateClient({ authMode: 'userPool' });
```

### Critical: Custom Queries

Auto-generated queries fetch nested relations which cause `@auth` errors. Always create `src/graphql/customQueries.js` and `src/graphql/customMutations.js` with **flattened** queries:

```javascript
// customQueries.js
export const listItemsSimple = /* GraphQL */ `
  query ListItems($filter: ModelItemFilterInput, $limit: Int) {
    listItems(filter: $filter, limit: $limit) {
      items {
        id
        name
        description
        imageKey
        createdAt
        updatedAt
        owner
        __typename
      }
      nextToken
    }
  }
`;
```

Use them in your context:

```typescript
const result = await client.graphql({
    query: listItemsSimple,
    variables: { filter: { ... }, limit: 100 }
});
const items = (result as any).data.listItems.items;
```

## Step 10: Implement S3 Storage

See [references/storage-patterns.md](references/storage-patterns.md) for the complete pattern.

```typescript
import { uploadData, getUrl } from 'aws-amplify/storage';

// Upload
const response = await fetch(localUri);
const blob = await response.blob();
await uploadData({
    key: `images/${Date.now()}.jpg`,
    data: blob,
    options: { accessLevel: 'guest' }
}).result;

// Get signed URL
const { url } = await getUrl({
    key: imageKey,
    options: { validateObjectExistence: true, accessLevel: 'guest' }
});
```

## AppContext Pattern

Centralize all Amplify logic in a single React Context. See `assets/AppContext.tsx` for a complete template.

Key responsibilities:
- Auth state management with `Hub.listen('auth', ...)`
- Auto-create DB profile on first sign-in
- CRUD operations for all models
- S3 image upload/download helpers
- Loading state management

## Common Gotchas

1. **`@auth` errors on nested queries** — Always use custom flattened queries, never auto-generated ones with relations
2. **`signIn` when already signed in** — Catch `'already a signed in user'` error, call `signOut()` first, then retry
3. **S3 access levels** — Use `'guest'` for content shared between users
4. **Missing `react-native-get-random-values`** — Required by Amplify, import at top of entry file
5. **`Hub.listen` cleanup** — Always return the unsubscribe function in `useEffect` cleanup
6. **Android: missing `<uses-permission>`** — Ensure `INTERNET` permission in `AndroidManifest.xml`
7. **Amplify push fails** — Run `amplify status` to verify resources, then `amplify push --force` if needed
