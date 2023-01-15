# The purpose of this script is to perform the initial diarization (based on periods of low intensity within the .wav file). 
# If a pause is identified, the audio file is split at that location. 
# For each initial audio file (i.e., interview), the end result is a series of shorter audio files, the boundaries of which correspond with a 
# pause specified by the paramaters between lines 9-28

# Clear history
clearinfo

# Define parameters
form Give the parameters for pause analysis
   comment This script marks the pauses in the LongSound to the IntervalTier of the TextGrid.
   comment Give the time period you wish to include (The TextGrid will be overwritten!):
   real Starting_time_(seconds) 0
   real Finishing_time_(0=all) 0
   comment The following criteria define a pause:
   positive Minimum_duration_(seconds) 0.2
   positive Maximum_intensity_(dB) 49
   comment Give the intensity analysis parameters:
	 positive Minimum_pitch_(Hz) 100
	 integer Time_step_(0=auto) 0
   comment Give the window size for the intensity analyses (smaller window requires less memory):
	 positive Window_size_(seconds) 20
	 choice Boundary_placement 2
	button One boundary at the center of each pause
	button Two boundaries with a time margin of:
	positive Margin_(seconds) 0.1
	comment (The margin will not be used if the pause is shorter than 2 * margin.)
	boolean Mark_pause_intervals_with_xxx 1

  # Specify the location and extension of sound files
  comment Directory of sound files
  text sound_directory <URL FOR SOUND FILES>
  sentence Long_Sound_file_extension .wav


   comment Save TextGrid file to folder:
	text folder <LOCATION TO STORE TEXT GRID FILES>
endform


# -----------------------------------------------------
# Basic formatting
# -----------------------------------------------------

# Rule defining the minimum window length for pause analysis (in secs)
min_loop_len = 6.4 /  minimum_pitch


# -----------------------------------------------------
# Loop through sound files in directory
# -----------------------------------------------------

# Get a list of all sound files in the directory
strings = Create Strings as file list: "list", sound_directory$ + "*.wav"
numberOfFiles = Get number of strings



# Iterate through each sound file
for ifile to numberOfFiles
  selectObject: strings

  # Extract meta data
  filename$ = Get string: ifile
  soundname$ = filename$ - ".wav"

  # open WAV file as long sound file
  speechTurn = Open long sound file... 'sound_directory$''filename$'

  # create TextGrids
  speechTG = To TextGrid... speechTurn

  # check if TextGrid already exists
  #if fileReadable ("'folder$''soundname$'.TextGrid")
  #	pause The file 'folder$''soundname$'.TextGrid already exists. Do you want to overwrite it?
  #endif

  # -----------------------------------------------------
  # Define Pause Analysis Settings
  # -----------------------------------------------------

  # re-select LS file
  select LongSound 'soundname$'

  # Set some initial default parameters
	pausenumber = 0
	duration = 0
	count = 0
	loops = 0
	pauses_found = 0
	windowstart = 0
	windowend = 0
	frame = 0
	frames = 0
	time = 0
	intensity = 0
	pausedetected = 0
	pausestart = 0
	pauseend = 0
	pausenumber = 0
	halfpause = 0

  # Get length of each sound file
  selectObject: speechTG
  endofsound = Get end time


  # -----------------------------------------------------
  # Split sound file into shorter segments
  # -----------------------------------------------------

  duration = endofsound - starting_time

  # Set number of loops
  loops = 1
  if duration > window_size
    loops = ceiling ((duration/window_size))
  endif

  # Adjust number of loops if final loop < minimum length
  final_loop_duration = duration mod window_size

  if final_loop_duration < min_loop_len
    loops = loops-1
  endif


  count = 1
  latest_endboundary = 0

  # Create a window of the LongSound and extract it for analysis
  while count <= loops
  	windowstart = starting_time + ((count - 1) * window_size)
  	windowend = starting_time + (count * window_size)

    if windowend > endofsound
  	 windowend = endofsound
  	endif

#    if windowend > finishing_time
#  	 windowend = finishing_time
#  	endif

    selectObject: speechTurn
  	Extract part... windowstart windowend yes
  	windowname$ = "'soundname$'"+"_wind_" + "'count'" + "_of_" + "'loops'"
  	echo Analysing Intensity window 'count' of 'loops'
  	Rename... 'windowname$'

    # Calculate intensity
    To Intensity... minimum_pitch time_step
    frames = Get number of frames

    # Set  pause criteria
    pauseend = 0
    frame = 1

    # -----------------------------------------------------
    # Detect possible boundaries
    # -----------------------------------------------------
		# Loop through all frames in the Intensity object:
		while frame <= frames
			select Intensity 'windowname$'
			intensity = Get value in frame... frame
			time = Get time from frame... frame
				if intensity > maximum_intensity

					# If the end of an earlier detected possible pause has been reached:
					if pausedetected = 1
						if frame - 1 < 1
						pauseend = windowstart
						else
						pauseend = Get time from frame... (frame - 1)
						endif
						pausedetected = 0
					endif

				    # If below intensity limit, a possible new pause is started if one hasn't been detected yet:
				    elsif pausedetected = 0
						pausestart = Get time from frame... frame
						pauseend = 0
						pausedetected = 1
						pausenumber = pausenumber + 1
				# If a detected pause just continues, do nothing special.
				endif

      # -----------------------------------------------------
      # Place Boundaries
      # -----------------------------------------------------
			if pauseend > 0
				pauseduration = pauseend - pausestart
				if pauseduration >= minimum_duration
					select TextGrid 'soundname$'
					halfpause = pauseduration / 2
						if boundary_placement = 1
							boundary = pausestart + halfpause
							call BoundaryCheck
							if boundaryexists = 0
								Insert boundary... 1 boundary
								latest_endboundary = boundary
							endif
						else
							boundary = 0
							if pauseduration >= (2 * margin)
								if pausestart > margin
									boundary = pausestart + margin
									call BoundaryCheck
									if boundaryexists = 0 and boundary > latest_endboundary
										Insert boundary... 1 boundary
									endif
									#If the pause overlaps with the preceding pause, do a merge:
									if boundary = latest_endboundary
										Remove boundary at time... 1 boundary
									endif
								endif
								if mark_pause_intervals_with_xxx = 1
									pauseinterval = Get interval at time... 1 boundary
									Set interval text... 1 pauseinterval xxx
								endif
								boundary = pauseend - margin
								call BoundaryCheck
								if boundaryexists = 0 and boundary > latest_endboundary
									Insert boundary... 1 boundary
									latest_endboundary = boundary
								endif
							else
								if pauseend < (endofsound - margin)
									boundary = pausestart + halfpause
									call BoundaryCheck
									if boundaryexists = 0 and boundary > latest_endboundary
										Insert boundary... 1 boundary
										latest_endboundary = boundary
									endif
								endif
							endif
						endif
					pauseend = 0
					pauses_found = pauses_found + 1


					Write to text file... 'folder$''soundname$'.TextGrid
				endif
			endif
			frame = frame + 1
			# When all frames in the intensity analysis have been looked at, end the frame loop.
		endwhile

  # -----------------------------------------------------
  # Remove objects and increment counters
  # -----------------------------------------------------

	select Sound 'windowname$'
	Remove
	select Intensity 'windowname$'
	Remove

  # Increment loop counter
	count = count + 1

endwhile

# Write text grid
select TextGrid 'soundname$'
Write to text file... 'folder$''soundname$'.TextGrid

# echo Ready! The TextGrid file was saved as 'folder$''soundname$'.TextGrid.

endfor



#******************************************************************************************************

# internal function to check whether a boundary already exists
procedure BoundaryCheck
	tmpint = Get interval at time... 1 boundary
	tmpstart = Get starting point... 1 tmpint
	if tmpstart <> boundary
		boundaryexists = 0
	else
		boundaryexists = 1
	endif
endproc
