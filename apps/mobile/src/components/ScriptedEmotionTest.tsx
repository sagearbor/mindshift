/**
 * "Scripted test" card inside the Emotion dev tester. The phone asks for one
 * emotion at a time, listens for a few seconds, runs both models on that clip
 * and shows asked-vs-guessed. Uses its own microphone capture, so no Live
 * Coach session is needed (and none should be running).
 */
import { useCallback, useRef, useState } from "react";
import { StyleSheet, Text, TouchableOpacity, View } from "react-native";
import { requestRecordingPermissionsAsync, useAudioStream } from "expo-audio";
import type { AudioStreamBuffer } from "expo-audio";
import { setPlaybackMode, setRecordingMode } from "../utils/audioMode";
import { StreamingResampler, downmixToMono, float32ToInt16 } from "../utils/audio";
import { emotionDevHub } from "../emotion/hub";
import { SCRIPT_SENTENCE, TAKE_SECONDS, planTakes, summarize, tooQuiet, type ScoredTake } from "../emotion/scriptedTest";

const ROUNDS = 3;
const TARGET_RATE = 16000;
const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

type Phase =
  | { kind: "idle" }
  | { kind: "ready"; emotion: string; n: number; total: number }
  | { kind: "speak"; emotion: string; n: number; total: number }
  | { kind: "done" }
  | { kind: "error"; message: string };

export function ScriptedEmotionTest() {
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const [scored, setScored] = useState<ScoredTake[]>([]);
  const [note, setNote] = useState<string | null>(null);
  const collecting = useRef(false);
  const chunks = useRef<Int16Array[]>([]);
  const resampler = useRef<StreamingResampler | null>(null);
  const cancelled = useRef(false);

  const onBuffer = useCallback((buffer: AudioStreamBuffer) => {
    if (!collecting.current) return;
    let samples: Float32Array = new Float32Array(buffer.data);
    if (buffer.channels > 1) samples = downmixToMono(samples, buffer.channels);
    if (buffer.sampleRate !== TARGET_RATE) {
      if (!resampler.current || resampler.current.inputRate !== buffer.sampleRate) {
        resampler.current = new StreamingResampler(buffer.sampleRate, TARGET_RATE);
      }
      samples = resampler.current.process(samples);
    }
    chunks.current.push(float32ToInt16(samples));
  }, []);

  const { stream } = useAudioStream({ sampleRate: TARGET_RATE, channels: 1, encoding: "float32", onBuffer });

  const run = async () => {
    cancelled.current = false;
    setScored([]);
    setNote(null);
    try {
      if (!(await requestRecordingPermissionsAsync()).granted) {
        setPhase({ kind: "error", message: "Microphone permission was denied." });
        return;
      }
      if (!(await emotionDevHub.ensureRuntime())) {
        setPhase({ kind: "error", message: "Turn the tester switch ON above and make sure both models show ready, then try again." });
        return;
      }
      await setRecordingMode();
      await stream.start();
      const plan = planTakes(ROUNDS);
      const got: ScoredTake[] = [];
      for (let i = 0; i < plan.length && !cancelled.current; i++) {
        const take = plan[i];
        for (let attempt = 0; attempt < 2 && !cancelled.current; attempt++) {
          setPhase({ kind: "ready", emotion: take.expect, n: i + 1, total: plan.length });
          await sleep(2200);
          chunks.current = [];
          collecting.current = true;
          setPhase({ kind: "speak", emotion: take.expect, n: i + 1, total: plan.length });
          await sleep(TAKE_SECONDS * 1000);
          collecting.current = false;
          const total = chunks.current.reduce((a, c) => a + c.length, 0);
          const pcm = new Int16Array(total);
          let off = 0;
          for (const c of chunks.current) {
            pcm.set(c, off);
            off += c.length;
          }
          if (tooQuiet(pcm)) {
            setNote("That take was too quiet, asking again.");
            continue;
          }
          setNote(null);
          const r = await emotionDevHub.classifyClip(pcm, take.expect, take.round);
          if (r) {
            got.push({ expect: take.expect, tiny: r.tiny, small: r.small });
            setScored([...got]);
          }
          break;
        }
      }
      setPhase({ kind: "done" });
    } catch (e) {
      setPhase({ kind: "error", message: e instanceof Error ? e.message : String(e) });
    } finally {
      collecting.current = false;
      try {
        stream.stop();
      } catch {
        // already stopped
      }
      void setPlaybackMode().catch(() => {});
    }
  };

  const busy = phase.kind === "ready" || phase.kind === "speak";
  const sum = scored.length ? summarize(scored) : null;

  return (
    <View style={styles.box} testID="scripted-emotion-test">
      <Text style={styles.title}>Scripted test</Text>
      <Text style={styles.sub}>
        {`Answers "how good are the models on acted emotion?" in about 2 minutes. Stop any Live Coach session first. For each prompt say "${SCRIPT_SENTENCE}" in that emotion, then keep going in that voice for a few seconds. ${ROUNDS} rounds x 6 emotions.`}
      </Text>
      {phase.kind === "ready" ? (
        <Text style={styles.big}>{`Get ready: ${phase.emotion.toUpperCase()}  (${phase.n}/${phase.total})`}</Text>
      ) : null}
      {phase.kind === "speak" ? (
        <Text style={[styles.big, styles.go]}>{`SPEAK NOW: ${phase.emotion.toUpperCase()}  (${phase.n}/${phase.total})`}</Text>
      ) : null}
      {phase.kind === "error" ? <Text style={styles.err}>{phase.message}</Text> : null}
      {note ? <Text style={styles.sub}>{note}</Text> : null}
      <View style={styles.row}>
        <TouchableOpacity testID="scripted-test-start" disabled={busy} style={[styles.btn, busy && styles.off]} onPress={() => void run()}>
          <Text style={styles.btnText}>{scored.length ? "Run again" : "Start scripted test"}</Text>
        </TouchableOpacity>
        {busy ? (
          <TouchableOpacity style={styles.btnGrey} onPress={() => (cancelled.current = true)}>
            <Text style={styles.btnText}>Stop</Text>
          </TouchableOpacity>
        ) : null}
      </View>
      {sum ? (
        <View testID="scripted-test-results">
          <Text style={styles.big}>{`TINY right ${sum.tinyRight}/${sum.total}   SMALL right ${sum.smallRight}/${sum.total}`}</Text>
          {sum.rows.map((r) => (
            <Text key={r.emotion} style={styles.sub}>
              {`${r.emotion}: tiny ${r.tinyRight}/${r.takes}${r.tinyUsually ? ` (says ${r.tinyUsually})` : ""} · small ${r.smallRight}/${r.takes}${r.smallUsually ? ` (says ${r.smallUsually})` : ""}`}
            </Text>
          ))}
          <Text style={styles.sub}>Every take is also in Export log (kind "scripted", with what you were asked).</Text>
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  box: { borderWidth: 1, borderColor: "#d1d5db", borderRadius: 10, padding: 12, marginTop: 12, gap: 6 },
  title: { fontSize: 16, fontWeight: "700", color: "#111827" },
  sub: { fontSize: 13, color: "#4b5563" },
  big: { fontSize: 18, fontWeight: "700", color: "#111827", marginTop: 4 },
  go: { color: "#b91c1c" },
  err: { fontSize: 14, color: "#b91c1c" },
  row: { flexDirection: "row", gap: 8, marginTop: 6 },
  btn: { backgroundColor: "#2563eb", paddingVertical: 10, paddingHorizontal: 14, borderRadius: 8 },
  btnGrey: { backgroundColor: "#6b7280", paddingVertical: 10, paddingHorizontal: 14, borderRadius: 8 },
  off: { opacity: 0.5 },
  btnText: { color: "#fff", fontWeight: "700" },
});
