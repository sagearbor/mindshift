package app.gauge.shared.sentinel

/**
 * A circular byte buffer that overwrites the oldest data when capacity is exceeded.
 * Used for storing pre-trigger PCM audio samples with a fixed memory footprint.
 */
class RingBuffer(val capacityBytes: Int) {
    private val buffer = ByteArray(capacityBytes)
    private var head = 0  // index of oldest element
    private var size = 0  // number of bytes currently stored

    /**
     * Adds a chunk of bytes to the buffer.
     * If the chunk is larger than capacity, only the tail (last `capacityBytes`) are kept.
     * If the buffer is full, oldest bytes are overwritten.
     */
    fun push(chunk: ByteArray) {
        if (chunk.isEmpty() || capacityBytes == 0) return

        // If chunk is larger than capacity, keep only the tail
        val toAdd = if (chunk.size > capacityBytes) {
            chunk.copyOfRange(chunk.size - capacityBytes, chunk.size)
        } else {
            chunk
        }

        // Add toAdd bytes to the buffer
        for (byte in toAdd) {
            if (size < capacityBytes) {
                // Buffer not full yet
                buffer[(head + size) % capacityBytes] = byte
                size++
            } else {
                // Buffer is full, overwrite oldest
                buffer[head] = byte
                head = (head + 1) % capacityBytes
            }
        }
    }

    /**
     * Returns a snapshot of the buffer contents in order (oldest to newest).
     */
    fun snapshot(): ByteArray {
        if (size == 0) return byteArrayOf()

        val result = ByteArray(size)
        for (i in 0 until size) {
            result[i] = buffer[(head + i) % capacityBytes]
        }
        return result
    }

    /**
     * Clears the buffer, resetting size to 0.
     */
    fun clear() {
        head = 0
        size = 0
    }

    /**
     * Returns the current number of bytes stored in the buffer.
     */
    val sizeBytes: Int
        get() = size
}
