# Screen Patterns

Reusable layout patterns for common screen types.

## Dashboard Pattern

- **Header:** Logo left, avatar button right (bordered circle)
- **Hero card:** Title + subtitle left, circular action button right (with glow)
- **Grid:** 2-column grid of touchable cards with `icon + label`
- **Sections:** Uppercase section title + content cards

## Auth Screen Pattern

- **Layout:** Centered with `KeyboardAvoidingView` + `ScrollView`
- **Logo** at top, subtitle below
- **Inputs:** Stacked input groups with uppercase labels
- **Primary button:** Full width, rounded, with glow effect
- **Toggle link** at bottom to switch between Login/SignUp

## List Screen Pattern

- **Container:** `FlatList` or `ScrollView` with card items
- **Card layout:** Avatar left, info center, chevron right
- **Interaction:** `Pressable` with pressed state feedback (opacity + border highlight)

## Detail Screen Pattern

- **Hero image** or icon at top (full width or centered circle)
- **Content:** Card sections with label + value pairs
- **Actions:** Bottom-anchored button group (primary + secondary)

## Settings Screen Pattern

- **Sections:** Grouped list items with section headers
- **Row layout:** Icon left, label center, chevron or toggle right
- **Destructive actions:** Red-tinted at bottom (e.g., "Sign Out", "Delete Account")
