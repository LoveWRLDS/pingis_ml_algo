import { ColorSchemeName } from 'react-native';

// ============================================================
// CUSTOMIZE: Change these values to match your brand
// ============================================================
const palette = {
    // Primary accent — the hero color of your app
    accent: '#C9963A',      // Gold (change to your brand color)

    // Semantic colors for status indicators
    green: '#10B981',
    yellow: '#F59E0B',
    red: '#EF4444',
    orange: '#F97316',
    blue: '#3B82F6',
};

const lightTheme = {
    background: '#FFFFFF',
    foreground: '#3D3530',
    border: '#E5DDD6',
    card: '#F5F5F5',
    accent: palette.accent,
    text: {
        green: '#059669',
        yellow: '#D97706',
        red: '#DC2626',
        orange: '#EA580C',
    },
    brand: palette,
    spacing: { xs: 4, sm: 8, md: 16, lg: 24, xl: 32, xxl: 48 },
    fontSize: { xs: 10, sm: 12, base: 16, lg: 20, xl: 24, xxl: 32, jumbo: 48 },
    radius: { sm: 4, md: 8, lg: 16, xl: 24, full: 9999 },
    shadows: {
        soft: {
            shadowColor: '#000',
            shadowOffset: { width: 0, height: 2 },
            shadowOpacity: 0.1,
            shadowRadius: 4,
            elevation: 2,
        }
    }
};

const darkTheme = {
    // CUSTOMIZE: Dark mode palette
    background: '#1A1512',  // Warm dark background
    foreground: '#F5EDE5',  // Warm light text
    border: '#A89F91',      // Subtle border / muted text
    card: '#2A221D',        // Card / surface color
    accent: palette.accent,
    text: {
        green: '#34D399',
        yellow: '#FBBF24',
        red: '#F87171',
        orange: '#FB923C',
    },
    brand: palette,
    spacing: { xs: 4, sm: 8, md: 16, lg: 24, xl: 32, xxl: 48 },
    fontSize: { xs: 10, sm: 12, base: 16, lg: 20, xl: 24, xxl: 32, jumbo: 48 },
    radius: { sm: 4, md: 8, lg: 16, xl: 24, full: 9999 },
    shadows: {
        soft: {
            shadowColor: '#000',
            shadowOffset: { width: 0, height: 4 },
            shadowOpacity: 0.3,
            shadowRadius: 8,
            elevation: 4,
        },
        glow: (color: string) => ({
            shadowColor: color,
            shadowOffset: { width: 0, height: 0 },
            shadowOpacity: 0.7,
            shadowRadius: 10,
            elevation: 8,
        }),
    },
};

export const spacing = {
    xs: 4,
    sm: 8,
    md: 16,
    lg: 24,
    xl: 32,
    xxl: 48,
};

export const fontSize = {
    xs: 12,
    sm: 14,
    base: 16,
    lg: 18,
    xl: 20,
    xxl: 24,
    jumbo: 32,
};

export const radius = {
    sm: 4,
    md: 8,
    lg: 12,
    xl: 16,
    full: 9999,
};

export const getTheme = (scheme: ColorSchemeName) => {
    // CUSTOMIZE: Set to false to allow system light/dark toggle
    const forceDarkMode = true;

    const isDark = forceDarkMode || scheme === 'dark';
    const theme = isDark ? darkTheme : lightTheme;

    return {
        ...theme,
        brand: palette,
        spacing,
        fontSize,
        radius,
        isDark,
    };
};

export type Theme = ReturnType<typeof getTheme>;
