import React from "react";
import { SafeAreaView, StyleSheet } from "react-native";
import SessionScreen from "./src/screens/SessionScreen";

export default function App() {
  return (
    <SafeAreaView style={styles.container}>
      <SessionScreen />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F9FAFB",
  },
});