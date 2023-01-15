# A post-analysis step. The purpose of this function is to check whether all relevant IPUs have been correctly labelled by the raters.

clearinfo

form IPU Label Check

# Specify the location and extension of TextGrids
comment Directory of TextGrid files
text tg_directory <URL>

# Set path of export file
comment Full path of resulting csv file:
text resultdir <URL>
endform


# -----------------------------------------------------
# Basic formatting
# -----------------------------------------------------

# Set delimiter
sep$ = tab$


# -----------------------------------------------------
# Create output file
# -----------------------------------------------------
output$ = resultdir$ + "intervals.csv"

output_head$ =  "interview" + sep$
  ...+ "speaker" + sep$
  ...+ "turn" + sep$
  ...+ "total_intervals" + sep$
  ...+ "s_intervals" + sep$
  ...+ "o_intervals" + sep$
  ...+ "z_intervals" + sep$
  ...+ "n_intervals" + sep$
  ...+ "i_intervals" + sep$
  ...+ "xxx_intervals" + sep$
  ...+ "xxxLong_intervals" + sep$
  ...+ "blank_intervals" + sep$
  ...+ "missing_intervals" + sep$
  ...+ "first_interval_label" + sep$
  ...+ "last_interval_label" + sep$
  ...+ "first_interval_duration" + sep$
  ...+ "last_interval_duration" + sep$

writeFile:"'output$'"
writeFileLine: "'output$'", output_head$


# -----------------------------------------------------
# Loop through sound files in directory
# -----------------------------------------------------

# Get a list of all sound files in the directory
strings = Create Strings as file list: "list", tg_directory$ + "*.TextGrid"
numberOfFiles = Get number of strings


# Iterate through each sound file
for ifile to numberOfFiles
  selectObject: strings

  # Extract some basic data
  filename$ = Get string: ifile
	interview$ = left$(filename$, 5)
	speaker$ = mid$(filename$, 7,2)
  soundname$ = filename$ - ".TextGrid"
  turn$ = mid$(soundname$, 11,2)

  # open WAV file as long sound file
  tg = Read from file... 'tg_directory$''filename$'

  # Standardize labels
  Replace interval texts: 1, 1, 0, "S", "s", "Literals"
  Replace interval texts: 1, 1, 0, "O", "o", "Literals"
  Replace interval texts: 1, 1, 0, "Z", "z", "Literals"
  Replace interval texts: 1, 1, 0, "N", "n", "Literals"
  Replace interval texts: 1, 1, 0, "XXX", "xxx", "Literals"



  # loop through each xxx label and check length
  numIntervals = Get number of intervals: 1

  for i to numIntervals

    # find interval label
    ilab$ = Get label of interval: 1, i

    if ilab$ = "xxx"
      # get start and end time of interval
      start = Get start time of interval: 1, i
      end = Get end time of interval: 1, i

      # compute duration
      duration = end - start

      # if duration > threshold, change label
      if duration > 3
        Replace interval texts: 1, i, i, "xxx", "xxx_long", "Literals"
      endif

    endif

  endfor


  # now do the same for blank intervals
  for i to numIntervals

    # find interval label
    ilab$ = Get label of interval: 1, i

    if ilab$ = ""
      # get start and end time of interval
      start = Get start time of interval: 1, i
      end = Get end time of interval: 1, i

      # compute duration
      duration = end - start

      # if duration > threshold, change label
      if duration > 1
        Replace interval texts: 1, i, i, "", "missing", "Literals"
      endif

    endif

  endfor


  # get total interval count
  total_intervals$ = Get number of intervals: 1
  total_intervals$ = total_intervals$ - " intervals"

  s_interval_count$ = Count intervals where: 1, "is equal to", "s"
  s_interval_count$ = s_interval_count$ - " intervals containing s"

  o_interval_count$ = Count intervals where: 1, "is equal to", "o"
  o_interval_count$ = o_interval_count$ - " intervals containing o"

  z_interval_count$ = Count intervals where: 1, "is equal to", "z"
  z_interval_count$ = z_interval_count$ - " intervals containing z"

  n_interval_count$ = Count intervals where: 1, "is equal to", "n"
  n_interval_count$ = n_interval_count$ - " intervals containing n"

  i_interval_count$ = Count intervals where: 1, "is equal to", "i"
  i_interval_count$ = i_interval_count$ - " intervals containing i"

  xxx_interval_count$ = Count intervals where: 1, "is equal to", "xxx"
  xxx_interval_count$ = xxx_interval_count$ - " intervals containing xxx"

  xxxLong_interval_count$ = Count intervals where: 1, "is equal to", "xxx_long"
  xxxLong_interval_count$ = xxxLong_interval_count$ - " intervals containing xxx_long"

  blank_interval_count$ = Count intervals where: 1, "is equal to", ""
  blank_interval_count$ = blank_interval_count$ - " intervals containing "

  missing_interval_count$ = Count intervals where: 1, "is equal to", "missing"
  missing_interval_count$ = missing_interval_count$ - " intervals containing missing"


  # Get label and duration for intervals at turn boundaries
  first_interval$ = Get label of interval: 1, 1
  start = Get start time of interval: 1, 1
  end = Get end time of interval: 1, 1
  # compute duration
  first_dur$ = string$ (end - start)

  last_interval$ = Get label of interval: 1, numIntervals
  start = Get start time of interval: 1, numIntervals
  end = Get end time of interval: 1, numIntervals
  # compute duration
  last_dur$ = string$ (end - start)




  # Output values
  output_vals$ = interview$ + sep$
    ...+ speaker$ + sep$
    ...+ turn$ + sep$
    ...+ total_intervals$ + sep$
    ...+ s_interval_count$ + sep$
    ...+ o_interval_count$ + sep$
    ...+ z_interval_count$ + sep$
    ...+ n_interval_count$ + sep$
    ...+ i_interval_count$ + sep$
    ...+ xxx_interval_count$ + sep$
    ...+ xxxLong_interval_count$ + sep$
    ...+ blank_interval_count$ + sep$
    ...+ missing_interval_count$ + sep$
    ...+ first_interval$ + sep$
    ...+ last_interval$ + sep$
    ...+ first_dur$ + sep$
    ...+ last_dur$ + sep$


  # Add to output file
  appendFileLine: "'output$'", output_vals$

  # remove TextGrid
  Remove

endfor


selectObject: strings
Remove
