# Schema Design Patterns

## Auth Rules

### Owner-only (private data)
```graphql
@auth(rules: [{ allow: owner }])
```

### Owner + authenticated read (profile-like data)
```graphql
@auth(rules: [
  { allow: owner },
  { allow: private, operations: [read] }
])
```

### Multi-owner read (shared access via viewers list)
```graphql
@auth(rules: [
  { allow: owner },
  { allow: owner, ownerField: "viewers", operations: [read] }
])
```
Add `viewers: [String]` field to the model. When sharing, push a user ID to the array.

### Multi-owner write (e.g. request approval by another party)
```graphql
@auth(rules: [
  { allow: owner, ownerField: "riderId" },
  { allow: owner, ownerField: "trainerId" }
])
```

## Relations

```graphql
# One-to-Many
type User @model {
  items: [Item] @hasMany
}

type Item @model {
  user: User @belongsTo
}
```

The auto-generated foreign key field is named `<parentType><parentField>Id`, e.g. `reinGaugeUserHorsesId`.

## Example: Full Schema Template

```graphql
type AppUser @model @auth(rules: [
  { allow: owner },
  { allow: private, operations: [read] }
]) {
  id: ID!
  email: String
  name: String
  avatarKey: String
  items: [AppItem] @hasMany
}

type AppItem @model @auth(rules: [
  { allow: owner },
  { allow: owner, ownerField: "viewers", operations: [read] }
]) {
  id: ID!
  name: String!
  description: String
  imageKey: String
  status: String
  user: AppUser @belongsTo
  viewers: [String]
  createdAt: AWSDateTime
}
```

## Viewers Pattern (Shared Access)

To share an item with another user:

```typescript
// Add viewer
const currentViewers = item.viewers || [];
const newViewers = [...new Set([...currentViewers, targetUserId])];
await client.graphql({
    query: updateItemSimple,
    variables: { input: { id: item.id, viewers: newViewers } }
});

// Remove viewer
const filteredViewers = (item.viewers || []).filter(id => id !== targetUserId);
await client.graphql({
    query: updateItemSimple,
    variables: { input: { id: item.id, viewers: filteredViewers } }
});
```
