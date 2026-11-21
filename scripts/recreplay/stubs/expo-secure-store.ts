// Node-side replay stub (see react-native.ts). Reads return null, writes drop.
export async function getItemAsync(_k: string): Promise<string | null> {
  return null;
}
export async function setItemAsync(_k: string, _v: string): Promise<void> {}
