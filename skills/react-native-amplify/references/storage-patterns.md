# S3 Storage Patterns

## Imports

```typescript
import { uploadData, getUrl } from 'aws-amplify/storage';
```

## Upload Image from Local URI

```typescript
const uploadImage = async (localUri: string, folder: string, filename: string): Promise<string> => {
    const response = await fetch(localUri);
    const blob = await response.blob();
    const key = `${folder}/${filename}-${Date.now()}.jpg`;

    await uploadData({
        key,
        data: blob,
        options: {
            accessLevel: 'guest',  // 'guest' for shared, 'private' for user-only
        }
    }).result;

    return key; // Store this key in your GraphQL model
};

// Usage:
const avatarKey = await uploadImage(imageUri, 'users', userId);
// Save avatarKey to DB via GraphQL mutation
```

## Get Signed URL (for display)

```typescript
const getSignedImageUrl = async (key: string): Promise<string | null> => {
    try {
        const result = await getUrl({
            key,
            options: {
                validateObjectExistence: true,
                accessLevel: 'guest',
            }
        });
        return result.url.toString();
    } catch (e) {
        console.log('Error signing image:', key, e);
        return null;
    }
};

// Usage in data fetching:
let avatarUrl = 'https://api.dicebear.com/7.x/initials/png?seed=User'; // Fallback
if (dbUser.avatarKey) {
    const signed = await getSignedImageUrl(dbUser.avatarKey);
    if (signed) avatarUrl = signed;
}
```

## Access Levels

| Level | Who can access | Use case |
|-------|---------------|----------|
| `guest` | Any authenticated user | Shared content (profile pics, public files) |
| `private` | Only the uploading user | Personal documents |
| `protected` | Owner: read/write, Others: read | Semi-public content |

## Key Naming Convention

Use a consistent folder structure:

```
users/{userId}-{timestamp}.jpg      — User avatars
items/{itemId}-{timestamp}.jpg      — Item images
uploads/{userId}/{filename}         — User uploads
```
