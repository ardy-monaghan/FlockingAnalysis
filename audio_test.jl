using VideoIO, ImageBinarization, Contour, CairoMakie, ImageFiltering, Statistics, PortAudio, Interpolations

function flock_outline(x_vec, y_vec, image_values)
    median_contour = Contour.levels(contours(x_vec,y_vec,image_values, 1))[1]
    line_vec = Contour.lines(median_contour)

    return coordinates(argmax(t -> length(t.vertices), line_vec))
end

function flock_midpoint(xs, ys)
    return ((minimum(xs) + maximum(xs)) / 2, (minimum(ys) + maximum(ys)) / 2)
end

function distance2centre(xs, ys)
    mx, my = flock_midpoint(xs, ys)
    dists = sqrt.((xs .- mx).^2 .+ (ys .- my).^2)
    return dists
end

function flock_deviations(xs, ys)
    dists = distance2centre(xs, ys)
    mean_dist = mean(dists)
    deviations = dists .- mean_dist
    return deviations
end


##
global const bpm = Ref{Float64}(120.0)
global const spf = Ref{Int64}(256) 
global const fs = Ref{Int64}(44100)

##
starlings = VideoIO.load("starlings.mp4")
clipped_starlings = starlings[1:600]

rough_alg = AdaptiveThreshold(window_size = 4; percentage = 1)
fine_alg = AdaptiveThreshold(window_size = 200; percentage = 0.01)

# Initialize audio signal storage
all_audio = Vector{Float64}[]

# Loop over every frame
for (i, raw_img) in enumerate(clipped_starlings)
    first_blur = imfilter(raw_img, Kernel.gaussian(1))
    img = binarize(first_blur, rough_alg)
    second_blur = imfilter(img, Kernel.gaussian(12))
    second_raster = binarize(second_blur, fine_alg)

    image_values = [float(second_raster[j, i].val) for j in axes(second_raster, 1), i in axes(second_raster, 2)]

    x_vec = 1:size(image_values, 1)
    y_vec = 1:size(image_values, 2)

    image_values .= reverse(image_values, dims=2)

    try
        (xs, ys) = flock_outline(x_vec, y_vec, image_values) # coordinates of this line segment
    catch e
        @warn "Could not extract flock outline for frame $i: $e"
        continue
    end

    (xs, ys) = flock_outline(x_vec, y_vec, image_values) # coordinates of this line segment

    audio_signal = flock_deviations(xs, ys)

    push!(all_audio, audio_signal)
end

## Process audio signal

# Phase alignment

# Scale the audio for volume
final_audio = Float64[]
largest_amplitude = maximum(signal -> maximum(abs.(signal)), all_audio)
scaled_audio = [signal ./ largest_amplitude for signal in all_audio]


# Interpolate to fixed length
fps = 30.0
samples_per_frame = floor(Int, fs[] / fps)

for signal in scaled_audio

    x = range(1, samples_per_frame, length(signal))
    y = signal

    itp_cubic = cubic_spline_interpolation(x, y)

    x_new = 1:samples_per_frame

    for x_val in x_new
        push!(final_audio, itp_cubic(x_val))
    end
    
end

## Test plotting

fig = Figure()
ax = Axis(fig[1, 1])

lines!(ax, 1:length(scaled_audio[1]), scaled_audio[1], color = :blue)
lines!(ax, 1:length(scaled_audio[2]), scaled_audio[2], color = :blue)
lines!(ax, 1:length(scaled_audio[3]), scaled_audio[3], color = :blue)

display(fig)

##
running = Ref(true)
stream = PortAudioStream(0, 1; samplerate=fs[])

##
@async begin
    try
        tᵢ = 0
        buf = zeros(Float32, spf[])

        while running[]

            write(stream, final_audio[tᵢ*spf[] .+ 1 : (tᵢ+1)*spf[]])
            tᵢ += 1

        end

    catch e
        @error "Audio task crashed" exception=(e, catch_backtrace())
    end
end

##
running[] = false
close(stream)