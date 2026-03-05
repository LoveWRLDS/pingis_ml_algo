# GraphQL CRUD Patterns

## Client Setup

Always use `userPool` auth mode for owner-based operations:

```typescript
import { generateClient } from 'aws-amplify/api';

const client = generateClient({ authMode: 'userPool' });
```

## Why Custom Queries

Auto-generated queries (`src/graphql/queries.js`) fetch **nested relations** which trigger `@auth` errors when the current user doesn't own the related records. Always create custom flattened queries.

## Custom Query Template

Create `src/graphql/customQueries.js`:

```javascript
export const listItemsSimple = /* GraphQL */ `
  query ListItems(
    $filter: ModelItemFilterInput
    $limit: Int
    $nextToken: String
  ) {
    listItems(filter: $filter, limit: $limit, nextToken: $nextToken) {
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
      __typename
    }
  }
`;

export const getItemSimple = /* GraphQL */ `
  query GetItem($id: ID!) {
    getItem(id: $id) {
      id
      name
      description
      imageKey
      createdAt
      updatedAt
      owner
      __typename
    }
  }
`;
```

## Custom Mutation Template

Create `src/graphql/customMutations.js`:

```javascript
export const createItemSimple = /* GraphQL */ `
  mutation CreateItem(
    $input: CreateItemInput!
    $condition: ModelItemConditionInput
  ) {
    createItem(input: $input, condition: $condition) {
      id
      name
      description
      imageKey
      createdAt
      updatedAt
      owner
      __typename
    }
  }
`;

export const updateItemSimple = /* GraphQL */ `
  mutation UpdateItem(
    $input: UpdateItemInput!
    $condition: ModelItemConditionInput
  ) {
    updateItem(input: $input, condition: $condition) {
      id
      name
      updatedAt
      owner
      __typename
    }
  }
`;

export const deleteItemSimple = /* GraphQL */ `
  mutation DeleteItem($input: DeleteItemInput!) {
    deleteItem(input: $input) {
      id
      __typename
    }
  }
`;
```

## CRUD Operations

### Create
```typescript
const input = { name: 'New Item', description: 'Details' };
const result = await client.graphql({
    query: createItemSimple,
    variables: { input }
});
const newItem = (result as any).data.createItem;
```

### Read (List with filter)
```typescript
const result = await client.graphql({
    query: listItemsSimple,
    variables: {
        filter: { name: { contains: 'search' } },
        limit: 100
    }
});
const items = (result as any).data.listItems.items;
```

### Read (Get by ID)
```typescript
const result = await client.graphql({
    query: getItemSimple,
    variables: { id: itemId }
});
const item = (result as any).data.getItem;
```

### Update
```typescript
const result = await client.graphql({
    query: updateItemSimple,
    variables: { input: { id: itemId, name: 'Updated Name' } }
});
```

### Delete
```typescript
await client.graphql({
    query: deleteItemSimple,
    variables: { input: { id: itemId } }
});
```

### Delete Error Handling

Amplify sometimes throws errors even on successful deletions. Handle gracefully:

```typescript
try {
    await client.graphql({
        query: deleteItemSimple,
        variables: { input: { id } }
    });
    removeFromLocalState(id);
} catch (e: any) {
    if (e?.data?.deleteItem) {
        // Deletion succeeded despite error
        removeFromLocalState(id);
    } else {
        console.error('Delete failed:', e);
    }
}
```

## Foreign Key Pattern

When creating a child item linked to a parent, use the auto-generated FK field:

```typescript
// FK field name: <parentModelName><parentField>Id (camelCase)
// Example: User.items -> appUserItemsId
const input = {
    name: 'Child Item',
    appUserItemsId: userId,  // Links to parent User
};
```

## Pagination

For large datasets, handle `nextToken`:

```typescript
let allItems: any[] = [];
let nextToken: string | null = null;

do {
    const result: any = await client.graphql({
        query: listItemsSimple,
        variables: { limit: 100, nextToken }
    });
    allItems = [...allItems, ...result.data.listItems.items];
    nextToken = result.data.listItems.nextToken;
} while (nextToken);
```
