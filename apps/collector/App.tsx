import React from 'react';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { DataCollectionScreen } from './src/DataCollectionScreen';

export default function App() {
  return (
    <SafeAreaProvider>
      <DataCollectionScreen />
    </SafeAreaProvider>
  );
}
