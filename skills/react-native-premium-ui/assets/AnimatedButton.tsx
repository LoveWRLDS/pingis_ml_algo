import React, { useRef } from 'react';
import { Animated, TouchableOpacityProps, TouchableWithoutFeedback, ViewStyle, StyleProp } from 'react-native';

interface AnimatedButtonProps extends TouchableOpacityProps {
    style?: StyleProp<ViewStyle>;
    scaleTo?: number;
    duration?: number;
}

export const AnimatedButton: React.FC<AnimatedButtonProps> = ({
    children,
    style,
    scaleTo = 0.95,
    duration = 100,
    onPress,
    onPressIn,
    onPressOut,
    disabled,
    ...rest
}) => {
    const scaleAnimation = useRef(new Animated.Value(1)).current;

    const handlePressIn = (e: any) => {
        Animated.timing(scaleAnimation, {
            toValue: scaleTo,
            duration: duration,
            useNativeDriver: true,
        }).start();
        if (onPressIn) onPressIn(e);
    };

    const handlePressOut = (e: any) => {
        Animated.timing(scaleAnimation, {
            toValue: 1,
            duration: duration,
            useNativeDriver: true,
        }).start();
        if (onPressOut) onPressOut(e);
    };

    const animatedStyle = {
        transform: [{ scale: scaleAnimation }],
        opacity: disabled ? 0.6 : 1,
    };

    return (
        <TouchableWithoutFeedback
            onPress={disabled ? undefined : onPress}
            onPressIn={disabled ? undefined : handlePressIn}
            onPressOut={disabled ? undefined : handlePressOut}
            disabled={disabled}
            {...rest}
        >
            <Animated.View style={[style, animatedStyle]}>
                {children}
            </Animated.View>
        </TouchableWithoutFeedback>
    );
};
