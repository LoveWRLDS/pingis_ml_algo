---
name: react-native-premium-ui
description: Scaffold a premium React Native app with a dark-mode-first design system, reusable components, and navigation boilerplate. Use when creating a new React Native app that needs a polished, premium look with customizable colors, typography, spacing, and shadows. Triggers on requests like "create a new React Native app", "scaffold a mobile app", "set up a premium mobile UI", or when a dark-mode-first design skeleton is needed.
---

# React Native Premium UI Skeleton

Scaffold a production-quality React Native app with a dark-mode-first premium design system extracted from battle-tested production apps.

## Stack

- React Native (CLI, not Expo)
- React Navigation (Stack + Bottom Tabs)
- lucide-react-native for icons
- react-native-safe-area-context
- react-native-system-navigation-bar (Android immersive mode)

## Quick Start

1. Copy the template files from `assets/` into the new project's `src/` directory
2. Customize the theme tokens in `theme.ts` (colors, fonts, spacing)
3. Build screens using the `makeStyles(theme)` pattern
4. Wire up navigation in `App.tsx`

## Design System

### Theme Architecture

The theme is a centralized token system in `src/styles/theme.ts`. It provides:

- **Color palette** — accent/brand, semantic colors (green/yellow/red/orange)
- **Light + Dark themes** — with separate background, foreground, border, card colors
- **Spacing scale** — xs(4), sm(8), md(16), lg(24), xl(32), xxl(48)
- **Font size scale** — xs(12), sm(14), base(16), lg(18), xl(20), xxl(24), jumbo(32)
- **Border radius scale** — sm(4), md(8), lg(12), xl(16), full(9999)
- **Shadow presets** — `soft` shadow + dark-mode `glow(color)` function

Default is **forced dark mode** for a premium feel. To change, edit the `getTheme()` function.

### Customization Points

When creating a new app from this skeleton, customize these values:

| Token | Default | What to change |
|-------|---------|----------------|
| `palette.gold` | `#C9963A` | Main accent/brand color |
| `darkTheme.background` | `#1A1512` | Dark bg (warm brown-black) |
| `darkTheme.foreground` | `#F5EDE5` | Light text on dark bg |
| `darkTheme.card` | `#2A221D` | Card/surface color |
| `darkTheme.border` | `#A89F91` | Border/subtle text |
| Semantic colors | green/yellow/red/orange | Status indicators |

### Key Patterns

#### 1. `makeStyles(theme)` Factory

Every screen and component uses this pattern for theme-aware styles:

```tsx
import { getTheme, Theme } from '../styles/theme';

const MyScreen = () => {
    const scheme = useColorScheme();
    const theme = getTheme(scheme);
    const styles = makeStyles(theme);
    // ...
};

const makeStyles = (theme: Theme) => StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: theme.background,
        padding: theme.spacing.lg,
    },
    title: {
        fontSize: theme.fontSize.xxl,
        fontWeight: 'bold',
        color: theme.brand.gold, // Accent color
        letterSpacing: 2,
    },
    card: {
        backgroundColor: theme.card,
        borderWidth: 1,
        borderColor: theme.border,
        borderRadius: theme.radius.lg,
        padding: theme.spacing.lg,
        ...theme.shadows.soft,
    },
});
```

#### 2. Typography Style

Premium typography uses uppercase labels, letter-spacing, and bold weights:

```tsx
// Section label
label: { color: theme.border, fontSize: 10, fontWeight: '700', letterSpacing: 1, textTransform: 'uppercase' }

// Screen title
title: { fontSize: 24, fontWeight: '900', color: theme.brand.gold, letterSpacing: 2 }

// Body text
body: { fontSize: theme.fontSize.base, color: theme.foreground, lineHeight: 20 }

// Muted/secondary
muted: { fontSize: theme.fontSize.sm, color: theme.foreground, opacity: 0.7 }
```

#### 3. Card Pattern

Cards use border + subtle shadow + rounded corners:

```tsx
card: {
    backgroundColor: theme.card,
    borderWidth: 1,
    borderColor: theme.border,
    borderRadius: theme.radius.lg,   // 12px
    padding: theme.spacing.lg,       // 24px
    ...theme.shadows.soft,
}
```

#### 4. Input Fields

Inputs are tall (56px), with card background and rounded corners:

```tsx
input: {
    height: 56,
    backgroundColor: theme.card,
    borderWidth: 1,
    borderColor: theme.border,
    borderRadius: 12,
    paddingHorizontal: 16,
    fontSize: 16,
    color: theme.foreground,
}
```

#### 5. Primary Button (with glow)

Buttons use accent colors with a glow shadow effect:

```tsx
primaryBtn: {
    height: 56,
    backgroundColor: theme.brand.gold,
    borderRadius: 28,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: theme.brand.gold,
    shadowOpacity: 0.4,
    shadowRadius: 10,
    elevation: 8,
}
```

#### 6. Pressable with Feedback

Use `Pressable` with `pressed` state for interactive cards:

```tsx
<Pressable style={({ pressed }) => [styles.card, pressed && styles.pressed]}>

pressed: {
    opacity: 0.9,
    borderColor: theme.brand.gold,
    backgroundColor: theme.brand.gold + '10',
}
```

#### 7. Icon Circles

Circular icon containers with accent color and glow:

```tsx
iconWrapper: {
    width: 64, height: 64, borderRadius: 32,
    backgroundColor: theme.brand.gold,
    alignItems: 'center', justifyContent: 'center',
    shadowColor: theme.brand.gold,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3, shadowRadius: 5, elevation: 5,
}
```

## App Structure

```
src/
├── styles/theme.ts          # Design tokens
├── components/
│   └── AnimatedButton.tsx   # Scale-on-press button
├── context/
│   └── AppContext.tsx        # Global state (auth, data)
├── navigation/
│   └── MainTabNavigator.tsx  # Bottom tab bar
├── screens/
│   ├── AuthScreen.tsx        # Login/SignUp
│   ├── Dashboard.tsx         # Home with hero card + grid
│   └── ...                   # Feature screens
└── assets/                   # Logo, icons
```

### Navigation Setup

- `App.tsx`: `SafeAreaProvider` → `AppProvider` → `NavigationContainer` → `Stack.Navigator`
- Auth guard: show auth screens when `!user`, app screens when authenticated
- Bottom tabs: use `createBottomTabNavigator` with theme-colored tab bar
- Tab bar styling: background matches theme, active tint = accent, inactive tint = border

### Tab Bar Style

```tsx
tabBarStyle: {
    backgroundColor: theme.background,
    borderTopColor: 'rgba(255,255,255,0.1)',
    elevation: 0,
    height: Platform.OS === 'ios' ? 88 : 60,
    paddingBottom: Platform.OS === 'ios' ? 28 : 8,
},
tabBarActiveTintColor: theme.brand.gold,
tabBarInactiveTintColor: theme.border,
tabBarLabelStyle: { fontSize: 10, fontWeight: '600' },
```

## Screen Templates

See [references/screen-patterns.md](references/screen-patterns.md) for complete screen layout patterns (Dashboard, Auth, List screens).

## AnimatedButton Component

A reusable button with scale-down animation on press:

```tsx
<AnimatedButton
    style={styles.myButton}
    onPress={handlePress}
    scaleTo={0.95}    // optional, default 0.95
    duration={100}    // optional, default 100ms
>
    <Text>Press Me</Text>
</AnimatedButton>
```

Uses React Native `Animated` API with `useNativeDriver: true` for smooth 60fps animations.

## Android Immersive Mode

For a truly premium full-screen experience on Android:

```tsx
import SystemNavigationBar from 'react-native-system-navigation-bar';

useEffect(() => {
    if (Platform.OS === 'android') {
        SystemNavigationBar.stickyImmersive();
    }
    StatusBar.setHidden(true);
}, []);
```
