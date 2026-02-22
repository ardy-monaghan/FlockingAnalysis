using BlobTracking, Images, VideoIO


# Get the tracks

path = "./bird_data/feb26/tt.mp4"
io   = VideoIO.open(path)
vid  = VideoIO.openvideo(io)
img  = first(vid)

# path = "./bird_data/feb26/test.mov"
# io   = VideoIO.open(path)
# vid  = VideoIO.openvideo(io)
# img  = first(vid)

medbg = MedianBackground(Float32.(img), 100) # A buffer of 100 frames
foreach(1:100) do i # Populate the buffer
    update!(medbg, Float32.(first(vid)))
end
bg = background(medbg)

Gray.(bg)

mask = ones(size(bg))
mask[1:270,:] .= 0

Gray.(mask)


function preprocessor(storage, img)
    storage .= Float32.(img)
    update!(medbg, storage) # update the background model
    storage .= Float32.(abs.(storage .- background(medbg)) .> 0.25) # You can save some computation by not calculating a new background image every sample
end

# Get coordinates
bt = BlobTracker(5:10, #sizes 
                2.0, # σw Dynamics noise std.
                10.0,  # σe Measurement noise std. (pixels)
                mask=mask,
                preprocessor = preprocessor,
                amplitude_th = 0.05,
                correspondence = HungarianCorrespondence(p=1.0, dist_th=2), # dist_th is the number of sigmas away from a predicted location a measurement is accepted.
)

##
# coords = get_coordinates(bt, vid)


result = track_blobs(bt, vid,
                         display = nothing, # use nothing to omit displaying.
                         recorder = Recorder()) # records result to video on disk


## Plot the tracks

traces = trace(result, minlife=40) # Filter minimum lifetime of 5
measurement_traces = tracem(result, minlife=5)
drawimg = RGB.(img)
draw!(drawimg, traces, c=RGB(0,0,0.5))
# draw!(drawimg, measurement_traces, c=RGB(0.5,0,0))


## Testing


result = TrackingResult()
buffer =  vid
ws = BlobTracking.Workspace(img, length(bt.sizes))
img, coord_or_img = img isa Tuple ? img : (img,img)
measurement = Measurement(ws, bt, coord_or_img, result)

# Update the results for each frame
for (ind,img) in enumerate(buffer)

    particle_shifted = Int64[]

    # Update the blobs
    println("Frame $ind")
    img, coord_or_img = img isa Tuple ? img : (img,img)
    measurement = update!(ws, bt, coord_or_img , result)

    # Check which blobs are left of the line
    # Loop over the blobs
    for (bᵢ, blob) in enumerate(result.blobs)
        if length(blob.trace) <= 1 
            continue
        end
        if blob.tracem[end-1][2] < 580 && blob.tracem[end][2] >= 580
            push!(particle_shifted, bᵢ)
            println("Particle shifts at $(ind)")
        end
    end

    # If position has switched play midi note
end

##


for (ind,img) in enumerate(buffer)
            println("Frame $ind")
            img, coord_or_img = img isa Tuple ? img : (img,img)
            measurement = update!(ws, bt, coord_or_img , result)
            showblobs(RGB.(Gray.(img)),result,measurement, rad=6, recorder=recorder, display=display, ignoreempty=ignoreempty)
        end


## Get the midi notes 

# Loop over each time
# Check if a point was previously to the left, and now to the right
# Find which note to play (mod the y position)
# Push midi note