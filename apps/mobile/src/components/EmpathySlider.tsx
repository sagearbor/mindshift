import React from "react";
import { View, Text, StyleSheet } from "react-native";
import Slider from "@react-native-community/slider";

interface EmpathySliderProps {
  value: number;
  onValueChange: (value: number) => void;
}

function getEmpathyLabel(value: number): string {
  if (value <= 20) return "Assertive";
  if (value <= 50) return "Balanced";
  if (value <= 80) return "Empathetic";
  return "Full Empathy";
}

export { getEmpathyLabel };

export default function EmpathySlider({
  value,
  onValueChange,
}: EmpathySliderProps) {
  return (
    <View style={styles.container}>
      <Text style={styles.label}>
        Empathy: {Math.round(value)} — {getEmpathyLabel(value)}
      </Text>
      <Slider
        testID="empathy-slider"
        style={styles.slider}
        minimumValue={0}
        maximumValue={100}
        step={1}
        value={value}
        onValueChange={onValueChange}
        minimumTrackTintColor="#4A90D9"
        maximumTrackTintColor="#D1D5DB"
        thumbTintColor="#4A90D9"
      />
      <View style={styles.endLabels}>
        <Text style={styles.endLabel}>Assertive</Text>
        <Text style={styles.endLabel}>Full Empathy</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginVertical: 12,
    paddingHorizontal: 16,
  },
  label: {
    fontSize: 16,
    fontWeight: "600",
    marginBottom: 8,
    color: "#1F2937",
  },
  slider: {
    width: "100%",
    height: 40,
  },
  endLabels: {
    flexDirection: "row",
    justifyContent: "space-between",
  },
  endLabel: {
    fontSize: 12,
    color: "#6B7280",
  },
});
