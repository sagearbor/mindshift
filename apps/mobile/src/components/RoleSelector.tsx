import React from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";

const ROLES = [
  "Husband / Wife",
  "Parent / Child",
  "Manager / Employee",
  "Therapist / Patient",
  "Friend / Friend",
] as const;

export { ROLES };

interface RoleSelectorProps {
  selectedRole: string;
  onSelect: (role: string) => void;
}

export default function RoleSelector({
  selectedRole,
  onSelect,
}: RoleSelectorProps) {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Select Role</Text>
      <View style={styles.grid}>
        {ROLES.map((role) => {
          const isSelected = selectedRole === role;
          return (
            <TouchableOpacity
              key={role}
              testID={`role-${role}`}
              style={[styles.roleButton, isSelected && styles.selected]}
              onPress={() => onSelect(role)}
              accessibilityRole="button"
              accessibilityState={{ selected: isSelected }}
            >
              <Text
                style={[styles.roleText, isSelected && styles.selectedText]}
              >
                {role}
              </Text>
            </TouchableOpacity>
          );
        })}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginVertical: 12,
    paddingHorizontal: 16,
  },
  title: {
    fontSize: 16,
    fontWeight: "600",
    marginBottom: 8,
    color: "#1F2937",
  },
  grid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
  },
  roleButton: {
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    backgroundColor: "#F9FAFB",
  },
  selected: {
    backgroundColor: "#4A90D9",
    borderColor: "#4A90D9",
  },
  roleText: {
    fontSize: 14,
    color: "#374151",
  },
  selectedText: {
    color: "#FFFFFF",
    fontWeight: "600",
  },
});
