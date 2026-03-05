import React, { createContext, useContext, useState, ReactNode, useEffect } from 'react';
import { getCurrentUser, fetchUserAttributes, signOut } from 'aws-amplify/auth';
import { generateClient } from 'aws-amplify/api';
import { getUrl, uploadData } from 'aws-amplify/storage';
import { Hub } from 'aws-amplify/utils';

// CUSTOMIZE: Import your custom queries and mutations
// import { getUserSimple, listItemsSimple } from '../graphql/customQueries';
// import { createUserSimple, updateUserSimple, createItemSimple } from '../graphql/customMutations';

// Use User Pool auth for owner-based operations
const client = generateClient({ authMode: 'userPool' });

// ============================================
// CUSTOMIZE: Define your types
// ============================================
interface User {
    id: string;
    email: string;
    name: string;
    avatarUrl?: string;
    avatarKey?: string;
    // Add role, custom fields etc.
}

// CUSTOMIZE: Add your data types
// interface Item { id: string; name: string; ... }

// ============================================
// Context Interface
// ============================================
interface AppContextType {
    user: User | null;
    isLoading: boolean;
    signOutUser: () => void;
    refreshUser: () => Promise<void>;
    // CUSTOMIZE: Add your CRUD operations
    // items: Item[];
    // addItem: (name: string) => Promise<void>;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

// ============================================
// Provider
// ============================================
export const AppProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
    const [user, setUser] = useState<User | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    // CUSTOMIZE: Add your data state
    // const [items, setItems] = useState<Item[]>([]);

    // ---- S3 Helper ----
    const getSignedImage = async (key: string): Promise<string | null> => {
        try {
            const result = await getUrl({
                key,
                options: { validateObjectExistence: true, accessLevel: 'guest' }
            });
            return result.url.toString();
        } catch (e) {
            console.log('[AppContext] Error signing image:', key, e);
            return null;
        }
    };

    // ---- Auth ----
    const checkUser = async () => {
        setIsLoading(true);
        try {
            const authUser = await getCurrentUser();
            const attributes = await fetchUserAttributes();

            // 1. Set base user immediately (optimistic)
            const baseUser: User = {
                id: authUser.userId,
                email: attributes.email || '',
                name: attributes.name || 'User',
            };
            setUser(baseUser);

            // 2. Fetch/create DB profile (non-critical)
            try {
                // CUSTOMIZE: Replace with your query
                // let dbUser = await client.graphql({ query: getUserSimple, variables: { id: authUser.userId } });
                // if (!dbUser) { /* Create profile */ }
                // Resolve avatar, merge data, etc.
            } catch (dbError) {
                console.error('DB profile sync failed:', dbError);
                // Don't log out — user is authenticated
            }
        } catch (e) {
            setUser(null);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        checkUser();

        const listener = Hub.listen('auth', (data) => {
            if (data.payload.event === 'signedIn') checkUser();
            else if (data.payload.event === 'signedOut') {
                setUser(null);
                // CUSTOMIZE: Clear data state
            }
        });

        return () => listener();
    }, []);

    const signOutUser = async () => { await signOut(); };
    const refreshUser = async () => { await checkUser(); };

    // ---- CUSTOMIZE: CRUD Operations ----
    // const addItem = async (name: string) => { ... };

    // ---- S3 Upload Helper ----
    const uploadImage = async (localUri: string, folder: string, name: string): Promise<string> => {
        const response = await fetch(localUri);
        const blob = await response.blob();
        const key = `${folder}/${name}-${Date.now()}.jpg`;
        await uploadData({ key, data: blob, options: { accessLevel: 'guest' } }).result;
        return key;
    };

    return (
        <AppContext.Provider value={{
            user, isLoading, signOutUser, refreshUser,
            // CUSTOMIZE: expose your data & operations
        }}>
            {children}
        </AppContext.Provider>
    );
};

// ============================================
// Hook
// ============================================
export const useApp = () => {
    const context = useContext(AppContext);
    if (!context) throw new Error('useApp must be used within AppProvider');
    return context;
};
