import React, { useEffect } from 'react';
import { useColorScheme, StatusBar, Platform, View, ActivityIndicator } from 'react-native';
import { NavigationContainer, DefaultTheme, DarkTheme } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import SystemNavigationBar from 'react-native-system-navigation-bar';

// CUSTOMIZE: Import your screens and navigator
// import MainTabNavigator from './src/navigation/MainTabNavigator';
// import AuthScreen from './src/screens/AuthScreen';

import { getTheme } from './src/styles/theme';

// CUSTOMIZE: Define your route params
export type RootStackParamList = {
    // AuthScreen: undefined;
    MainTabs: undefined;
    // Add your routes here
};

const Stack = createNativeStackNavigator<RootStackParamList>();

const NavigationWrapper = () => {
    const scheme = useColorScheme();
    const theme = getTheme(scheme);

    // CUSTOMIZE: Add auth guard logic
    // const { user, isLoading } = useApp();

    return (
        <NavigationContainer
            theme={scheme === 'dark' ? DarkTheme : DefaultTheme}
            onReady={() => {
                if (Platform.OS === 'android') SystemNavigationBar.stickyImmersive();
            }}
        >
            <Stack.Navigator screenOptions={{ headerShown: false }}>
                {/* CUSTOMIZE: Add your screens */}
                {/* <Stack.Screen name="MainTabs" component={MainTabNavigator} /> */}
            </Stack.Navigator>
        </NavigationContainer>
    );
};

function App(): React.JSX.Element {
    useEffect(() => {
        const enableImmersive = async () => {
            if (Platform.OS === 'android') {
                await SystemNavigationBar.stickyImmersive();
            }
            StatusBar.setHidden(true);
        };
        enableImmersive();
    }, []);

    return (
        // CUSTOMIZE: Wrap with your providers (AppProvider, etc.)
        <SafeAreaProvider>
            <StatusBar hidden={true} />
            <NavigationWrapper />
        </SafeAreaProvider>
    );
}

export default App;
