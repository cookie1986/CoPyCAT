# Script: looks up the label of the first and last IPU. If either contains
# an incorrect value, then the adjacent IPU is checked. Once an appropriate
# start-end IPU is located, the script then performs a pause analysis on each,
# and extracts features based on intensity, pitch, and vocal quality.
# The output is saved as csv files.
# 12/08/21

# UPDATE: Version four reformats and simplifies version 3.

# Clear history
clearinfo



# -----------------------------------------------------
# Specify location of audio and textgrids
# -----------------------------------------------------

form IPU Label Check and Feature Extraction

  # Specify the location and extension of sound files
  comment Directory of sound files
  text sound_directory <URL>
	sentence Long_Sound_file_extension .wav

  # Specify the location and extension of sound files
  comment Directory of TextGrid files
  text tg_directory <URL>

  # Set path of csv file with feature values
  comment Full path of resulting csv file:
  text resultdir <URL>
endform




# -----------------------------------------------------
# Basic formatting
# -----------------------------------------------------

# Set delimiter
sep$ = tab$


# -----------------------------------------------------
# Create output files
# -----------------------------------------------------

# Output csv file
# Check if intensity results file exists:
output$ = resultdir$ + "praat_output.csv"
if fileReadable (output$)
	pause The result file 'output$' already exists! Overwrite?
	filedelete 'output$'
endif

# If no results file exists, write new file to store results
out_head$ =  "interview" + sep$
  ...+ "speaker" + sep$
  ...+ "turn" + sep$
  ...+ "turn_length" + sep$
  ...+ "segment_number" + sep$
  ...+ "segment_label" + sep$
  ...+ "ipu1_duration" + sep$
  ...+ "ipu2_duration" + sep$
  ...+ "int1_mean" + sep$
  ...+ "int1_max" + sep$
  ...+ "int1_sd" + sep$
  ...+ "int2_mean" + sep$
  ...+ "int2_max" + sep$
  ...+ "int2_sd" + sep$
  ...+ "pitch1_mean" + sep$
  ...+ "pitch1_max" + sep$
  ...+ "pitch1_sd" + sep$
  ...+ "pitch2_mean" + sep$
  ...+ "pitch2_max" + sep$
  ...+ "pitch2_sd" + sep$
  ...+ "jitter1" + sep$
  ...+ "shimmer1" + sep$
  ...+ "hnr1" + sep$
  ...+ "jitter2" + sep$
  ...+ "shimmer2" + sep$
  ...+ "hnr2" + sep$

writeFile:"'output$'"
writeFileLine: "'output$'", out_head$


# -----------------------------------------------------
# Loop through sound files in directory
# -----------------------------------------------------

# Get a list of all sound files in the directory
strings = Create Strings as file list: "list", sound_directory$ + "*.wav"
numberOfFiles = Get number of strings

# For Loop 1 - File/Interview Loop
# Iterate through each sound file
for ifile to numberOfFiles
  selectObject: strings

  # Basic Formatting
  filename$ = Get string: ifile
  interview$ = left$(filename$, 5)
  speaker$ = mid$(filename$, 7,2)
  soundname$ = filename$ - ".wav"
  turn$ = mid$(soundname$, 11,2)
  tg_file$ = filename$ - ".wav" + ".TextGrid"


  # open WAV file as long sound file
  speechTurn = Read from file... 'sound_directory$''filename$'

  # load the relevant (pre-labelled) text grid
  speechTG = Read from file... 'tg_directory$''tg_file$'

  # Get basic data regarding segments
  turn_length$ = Get total duration
  total_segments$ = Get number of intervals: 1
  correct_segments$ = Count intervals where: 1, "is equal to", "s"
  overlapped_segments$ = Count intervals where: 1, "is equal to", "o"
  incorrect_segments$ = Count intervals where: 1, "is equal to", "z"


  # -----------------------------------------------------
  # Loop through each segment in a TextGrid
  # -----------------------------------------------------

  # total segments in TextGrid
  total_intervals = Get number of intervals: 1

  # For Loop 2 - Segment Loop
  # Loop through each segment
  for seg_num to total_intervals

    # get duration of segment
    seg_start = Get start time of interval: 1, seg_num
    seg_end = Get end time of interval: 1, seg_num
    duration = seg_end - seg_start

    # set minimum segment length
    if duration > 1

      # store segment number as string
      segment$ = string$ (seg_num)

      # check segment label
      seg_lab$ = Get label of interval: 1, seg_num
      if seg_lab$ != "xxx" and seg_lab$ != "xxx_long" and seg_lab$ != "o" and seg_lab$ != "n" and seg_lab$ != "i" and seg_lab$ != ""

        # extract segment sound
        select Sound 'soundname$'
        seg_sound = Extract part: seg_start, seg_end, "rectangular", 1, "no"

        # reduce noise
        seg_sound_rednoi = Reduce noise: 0, 0, 0.025, 80, 10000, 40, -20, "spectral-subtraction"
        selectObject: seg_sound
        Remove
        selectObject: seg_sound_rednoi

        # Create intensity object
        intensity = To Intensity: 100, 0, "no"

        # Detect 50ms IPU boundaries
        ipu_TG = To TextGrid (silences): -25, 0.05, 0.1, "silence", "speech"
        selectObject: intensity
        Remove

        # -----------------------------------------------------
        # Extract first and last IPUs in segment
        # -----------------------------------------------------

        # First IPU in segment
        # Select IPU Text Grid
        selectObject: ipu_TG

        # total IPUs (incl. silence) in TextGrid
        total_ipu = Get number of intervals: 1

        # Conditional flag of ipu label (1 or 0)
        ipu_check = 1

        # Count IPUs from the beginning
        first_ipu_num = 1

        # check label of each IPU until "speech" is found
        while ipu_check = 1
          ipu_lab$ = Get label of interval: 1, first_ipu_num
          if ipu_lab$ = "speech"
            ipu_check = 0
          else
            # increment count
            first_ipu_num = first_ipu_num+1

            # set ipu_check to 1 if counter exceeds total IPUs
            if first_ipu_num > total_ipu

              ipu_check = 0
              first_ipu_num = 1
            endif
          endif
        endwhile

        # Extract sound for first eligible IPU
        first_IPU_start = Get start time of interval: 1, first_ipu_num
        first_IPU_end = Get end time of interval: 1, first_ipu_num

        selectObject: seg_sound_rednoi
        ipu1 = Extract part: first_IPU_start, first_IPU_end, "rectangular", 1, "no"


        # Last IPU in segment
        # re-select IPU Text Grid
        selectObject: ipu_TG

        # Conditional flag of ipu label (1 or 0)
        ipu_check = 1

        # Count IPUs from the end
        last_ipu_num = total_ipu

        # check label of each IPU until "speech" is found
        while ipu_check = 1
          ipu_lab$ = Get label of interval: 1, last_ipu_num
          if ipu_lab$ = "speech"
            ipu_check = 0
          else
            # increment count
            last_ipu_num = last_ipu_num-1

            # set ipu_check to 1 if counter exceeds total IPUs
            if last_ipu_num < 1

              ipu_check = 0
              last_ipu_num = 1
            endif
          endif
        endwhile

        # Extract sound for last eligible IPU
        last_IPU_start = Get start time of interval: 1, last_ipu_num
        last_IPU_end = Get end time of interval: 1, last_ipu_num

        selectObject: seg_sound_rednoi
        ipu2 = Extract part: last_IPU_start, last_IPU_end, "rectangular", 1, "no"

        # remove segment
        selectObject: seg_sound_rednoi, ipu_TG
        Remove

        # -----------------------------------------------------
        # Extract features
        # -----------------------------------------------------

        selectObject: ipu1

        # duration
        ipu1_dur$ = Get total duration

        # Intensity
        intensity1 = To Intensity: 100, 0, 1
        # Extract raw values
        intensity1_max$ = Get maximum: 0, 0, "Parabolic"
        intensity1_mean$ = Get mean: 0, 0, "dB"
        intensity1_sd$ = Get standard deviation: 0, 0
        # remove intensity object
        selectObject: intensity1
        Remove
        selectObject: ipu1

        # Pitch
        pitch1 = To Pitch: 0, 75, 600
        # Extract raw features from IPU
        pitch1_max$ = Get maximum: 0, 0, "Hertz", "Parabolic"
        pitch1_mean$ = Get mean: 0, 0, "Hertz"
        pitch1_sd$ = Get standard deviation: 0, 0, "Hertz"
        # remove pitch object
        selectObject: pitch1
        Remove
        selectObject: ipu1

        # VQ
        # create point process objects
        pp1 = To PointProcess (periodic, cc): 75, 600
        # Extract raw jitter and shimmer values
        jitter1$ = Get jitter (local): 0, 0, 0.0001, 0.02, 1.3
        selectObject: ipu1
        plusObject: pp1
        shimmer1$ = Get shimmer (local): 0, 0, 0.0001, 0.02, 1.3, 1.6
        # Extract HNR value
        selectObject: ipu1
        hnr1 = To Harmonicity (cc): 0.01, 60, 0.1, 1
        hnr1_mean$ = Get mean: 0, 0
        # remove non-needed files
        selectObject: pp1, hnr1
        Remove


        selectObject: ipu2

        # duration
        ipu2_dur$ = Get total duration

        # Intensity
        intensity2 = To Intensity: 100, 0, 1
        # Extract raw values
        intensity2_max$ = Get maximum: 0, 0, "Parabolic"
        intensity2_mean$ = Get mean: 0, 0, "dB"
        intensity2_sd$ = Get standard deviation: 0, 0
        # remove intensity object
        selectObject: intensity2
        Remove
        selectObject: ipu2

        # Pitch
        pitch2 = To Pitch: 0, 75, 600
        # Extract raw features from IPU
        pitch2_max$ = Get maximum: 0, 0, "Hertz", "Parabolic"
        pitch2_mean$ = Get mean: 0, 0, "Hertz"
        pitch2_sd$ = Get standard deviation: 0, 0, "Hertz"
        # remove pitch object
        selectObject: pitch2
        Remove
        selectObject: ipu2

        # VQ
        # create point process objects
        pp2 = To PointProcess (periodic, cc): 75, 600
        # Extract raw jitter and shimmer values
        jitter2$ = Get jitter (local): 0, 0, 0.0001, 0.02, 1.3
        selectObject: ipu2
        plusObject: pp2
        shimmer2$ = Get shimmer (local): 0, 0, 0.0001, 0.02, 1.3, 1.6
        # Extract HNR value
        selectObject: ipu2
        hnr2 = To Harmonicity (cc): 0.01, 60, 0.1, 1
        hnr2_mean$ = Get mean: 0, 0
        # remove non-needed files
        selectObject: pp2, hnr2
        Remove

        # -----------------------------------------------------
        # Build segment output
        # -----------------------------------------------------

        out_vals$ = interview$ + sep$
          ...+ speaker$ + sep$
          ...+ turn$ + sep$
          ...+ turn_length$ + sep$
          ...+ segment$ + sep$
          ...+ seg_lab$ + sep$
          ...+ ipu1_dur$ + sep$
          ...+ ipu2_dur$ + sep$
          ...+ intensity1_mean$ + sep$
          ...+ intensity1_max$ + sep$
          ...+ intensity1_sd$ + sep$
          ...+ intensity2_mean$ + sep$
          ...+ intensity2_max$ + sep$
          ...+ intensity2_sd$ + sep$
          ...+ pitch1_mean$ + sep$
          ...+ pitch1_max$ + sep$
          ...+ pitch1_sd$ + sep$
          ...+ pitch2_mean$ + sep$
          ...+ pitch2_max$ + sep$
          ...+ pitch2_sd$ + sep$
          ...+ jitter1$ + sep$
          ...+ shimmer1$ + sep$
          ...+ hnr1_mean$ + sep$
          ...+ jitter2$ + sep$
          ...+ shimmer2$ + sep$
          ...+ hnr2_mean$ + sep$

          # Add to file
          appendFileLine: "'output$'", out_vals$

          # Delete IPU sound files
          selectObject: ipu1, ipu2
          Remove

        # re-select the main TextGrid for the next segment
        selectObject: speechTG

      # endif for segment label check
      endif

    # endif for minimum segment length
    endif

  # End For Loop 2
  endfor

  # remove original speech turn and textgrid
  selectObject: speechTG, speechTurn
  Remove

# End For Loop 1
endfor


# Remove initial file list
selectObject: strings
Remove

echo All sound files successfully processed and saved
