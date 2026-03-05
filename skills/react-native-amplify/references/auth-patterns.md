# Auth Implementation Patterns

## Imports

```typescript
import { signIn, signUp, signOut, getCurrentUser, fetchUserAttributes, confirmSignUp } from 'aws-amplify/auth';
import { Hub } from 'aws-amplify/utils';
```

## Sign Up

```typescript
const handleSignUp = async (email: string, password: string, name: string) => {
    const { isSignUpComplete, nextStep } = await signUp({
        username: email,
        password,
        options: {
            userAttributes: {
                email,
                name,
                // Add custom attributes as needed:
                // profile: 'admin', // Store role in profile attribute
            }
        }
    });

    if (nextStep.signUpStep === 'CONFIRM_SIGN_UP') {
        // Navigate to email confirmation screen
    }
};
```

## Confirm Email

```typescript
const handleConfirm = async (email: string, code: string) => {
    await confirmSignUp({ username: email, confirmationCode: code });
    // Navigate to sign-in screen
};
```

## Sign In

```typescript
const handleSignIn = async (email: string, password: string) => {
    try {
        const { isSignedIn, nextStep } = await signIn({ username: email, password });
        if (isSignedIn) {
            // Refresh user state
        } else if (nextStep.signInStep === 'CONFIRM_SIGN_UP') {
            // Navigate to confirmation
        }
    } catch (error: any) {
        // Handle "already signed in" edge case
        if (error.message?.includes('already a signed in user')) {
            await signOut();
            // Retry signIn
        }
    }
};
```

## Check Current User (on app launch)

```typescript
const checkUser = async () => {
    try {
        const authUser = await getCurrentUser();
        const attributes = await fetchUserAttributes();

        const user = {
            id: authUser.userId,
            email: attributes.email || '',
            name: attributes.name || 'User',
            role: attributes.profile || 'user',
        };

        // Set user in state → unlocks app
        // Then fetch DB profile to enhance with stored data
    } catch (e) {
        // Not authenticated → show auth screen
    }
};
```

## Listen for Auth Events

```typescript
useEffect(() => {
    checkUser();

    const listener = Hub.listen('auth', (data) => {
        if (data.payload.event === 'signedIn') {
            checkUser();
        } else if (data.payload.event === 'signedOut') {
            setUser(null);
            // Clear data state
        }
    });

    return () => listener(); // Cleanup
}, []);
```

## Auth Guard in Navigation

```tsx
<Stack.Navigator screenOptions={{ headerShown: false }}>
    {!user ? (
        <>
            <Stack.Screen name="Auth" component={AuthScreen} />
            <Stack.Screen name="ConfirmEmail" component={ConfirmEmailScreen} />
        </>
    ) : (
        <>
            <Stack.Screen name="MainTabs" component={MainTabNavigator} />
            {/* ...other authenticated screens */}
        </>
    )}
</Stack.Navigator>
```

## Profile Sync Pattern

On first sign-in, the Cognito user exists but the DB profile may not. Auto-create it:

```typescript
// 1. Try to fetch DB profile
let dbUser = null;
try {
    const result = await client.graphql({
        query: getUserSimple,
        variables: { id: authUser.userId }
    });
    dbUser = result.data.getUser;
} catch (e) {}

// 2. If not found, create it
if (!dbUser) {
    const createResult = await client.graphql({
        query: createUserSimple,
        variables: { input: { id: authUser.userId, email, name } }
    });
    dbUser = createResult.data.createUser;
}
```
