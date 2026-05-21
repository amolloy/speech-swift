import AVFoundation
import CoreAudio
import os

/// Lock-free SPSC ring buffer for audio samples.
/// Producer (TTS thread) writes, consumer (audio render thread) reads.
public final class AudioSampleRingBuffer: @unchecked Sendable {
    private let buffer: UnsafeMutableBufferPointer<Float>
    private let capacity: Int
    private let lock = NSRecursiveLock()
    
    // Volatile indexes for thread safety
    private var readIndex: Int = 0
    private var writeIndex: Int = 0
    
    public var availableToRead: Int {
        (writeIndex - readIndex + capacity) % capacity
    }
    
    public var availableToWrite: Int {
        capacity - availableToRead - 1
    }
    
    public init(capacity: Int) {
        self.capacity = capacity
        let ptr = UnsafeMutablePointer<Float>.allocate(capacity: capacity)
        ptr.initialize(repeating: 0, count: capacity)
        self.buffer = UnsafeMutableBufferPointer(start: ptr, count: capacity)
    }
    
    deinit {
        buffer.baseAddress?.deallocate()
    }
    
    public func write(_ samples: [Float]) -> Int {
        let toWrite = min(samples.count, availableToWrite)
        for i in 0..<toWrite {
            buffer[(writeIndex + i) % capacity] = samples[i]
        }
        writeIndex = (writeIndex + toWrite) % capacity
        return toWrite
    }
    
    public func read(into output: UnsafeMutablePointer<Float>, count: Int) -> Int {
        let toRead = min(count, availableToRead)
        for i in 0..<toRead {
            output[i] = buffer[(readIndex + i) % capacity]
        }
        readIndex = (readIndex + toRead) % capacity
        return toRead
    }
}

public final class StreamingAudioPlayer {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private var sourceNode: AVAudioSourceNode?
    private let ringBuffer: AudioSampleRingBuffer
    private let lock = NSLock()
    
    private var playbackStarted = false
    private var generationComplete = false
    private let preBufferSamples: Int
    
    public init(sampleRate: Int = 24000, bufferDuration: Double = 10.0, preBufferDuration: Double = 0.5) {
        let bufferCapacity = Int(Double(sampleRate) * bufferDuration)
        let rb = AudioSampleRingBuffer(capacity: bufferCapacity)
        self.ringBuffer = rb
        self.preBufferSamples = Int(Double(sampleRate) * preBufferDuration)
        
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1, interleaved: false)!
        
        let node = AVAudioSourceNode(format: format) { [weak self] _, _, frameCount, bufferList -> OSStatus in
            guard let self = self else { return noErr }
            
            // Fixed for Mac Catalyst
            guard let mData = bufferList.pointee.mBuffers.mData else {
                return noErr
            }
            let dst = mData.assumingMemoryBound(to: Float.self)
            
            self.lock.lock()
            let started = self.playbackStarted
            let complete = self.generationComplete
            let available = rb.availableToRead
            self.lock.unlock()
            
            if !started {
                if available >= self.preBufferSamples || complete {
                    self.lock.lock()
                    self.playbackStarted = true
                    self.lock.unlock()
                } else {
                    for i in 0..<Int(frameCount) { dst[i] = 0 }
                    return noErr
                }
            }
            
            let read = rb.read(into: dst, count: Int(frameCount))
            if read < Int(frameCount) {
                for i in read..<Int(frameCount) { dst[i] = 0 }
            }
            
            return noErr
        }
        
        self.sourceNode = node
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: format)
    }
    
    public func start() throws {
        try engine.start()
    }
    
    public func stop() {
        engine.stop()
    }
    
    public func append(_ samples: [Float]) {
        _ = ringBuffer.write(samples)
    }
    
    public func setGenerationComplete() {
        lock.lock()
        generationComplete = true
        lock.unlock()
    }
}
