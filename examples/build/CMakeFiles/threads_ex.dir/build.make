# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/marc/workspace/dlib-master/examples

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/marc/workspace/dlib-master/examples/build

# Include any dependencies generated for this target.
include CMakeFiles/threads_ex.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/threads_ex.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/threads_ex.dir/flags.make

CMakeFiles/threads_ex.dir/threads_ex.cpp.o: CMakeFiles/threads_ex.dir/flags.make
CMakeFiles/threads_ex.dir/threads_ex.cpp.o: ../threads_ex.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/marc/workspace/dlib-master/examples/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/threads_ex.dir/threads_ex.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/threads_ex.dir/threads_ex.cpp.o -c /home/marc/workspace/dlib-master/examples/threads_ex.cpp

CMakeFiles/threads_ex.dir/threads_ex.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/threads_ex.dir/threads_ex.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/marc/workspace/dlib-master/examples/threads_ex.cpp > CMakeFiles/threads_ex.dir/threads_ex.cpp.i

CMakeFiles/threads_ex.dir/threads_ex.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/threads_ex.dir/threads_ex.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/marc/workspace/dlib-master/examples/threads_ex.cpp -o CMakeFiles/threads_ex.dir/threads_ex.cpp.s

CMakeFiles/threads_ex.dir/threads_ex.cpp.o.requires:
.PHONY : CMakeFiles/threads_ex.dir/threads_ex.cpp.o.requires

CMakeFiles/threads_ex.dir/threads_ex.cpp.o.provides: CMakeFiles/threads_ex.dir/threads_ex.cpp.o.requires
	$(MAKE) -f CMakeFiles/threads_ex.dir/build.make CMakeFiles/threads_ex.dir/threads_ex.cpp.o.provides.build
.PHONY : CMakeFiles/threads_ex.dir/threads_ex.cpp.o.provides

CMakeFiles/threads_ex.dir/threads_ex.cpp.o.provides.build: CMakeFiles/threads_ex.dir/threads_ex.cpp.o

# Object files for target threads_ex
threads_ex_OBJECTS = \
"CMakeFiles/threads_ex.dir/threads_ex.cpp.o"

# External object files for target threads_ex
threads_ex_EXTERNAL_OBJECTS =

threads_ex: CMakeFiles/threads_ex.dir/threads_ex.cpp.o
threads_ex: CMakeFiles/threads_ex.dir/build.make
threads_ex: dlib_build/libdlib.a
threads_ex: /usr/lib/x86_64-linux-gnu/libpthread.so
threads_ex: /usr/lib/x86_64-linux-gnu/libnsl.so
threads_ex: /usr/lib/x86_64-linux-gnu/libX11.so
threads_ex: /usr/lib/x86_64-linux-gnu/libpng.so
threads_ex: /usr/lib/x86_64-linux-gnu/libjpeg.so
threads_ex: CMakeFiles/threads_ex.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable threads_ex"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/threads_ex.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/threads_ex.dir/build: threads_ex
.PHONY : CMakeFiles/threads_ex.dir/build

CMakeFiles/threads_ex.dir/requires: CMakeFiles/threads_ex.dir/threads_ex.cpp.o.requires
.PHONY : CMakeFiles/threads_ex.dir/requires

CMakeFiles/threads_ex.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/threads_ex.dir/cmake_clean.cmake
.PHONY : CMakeFiles/threads_ex.dir/clean

CMakeFiles/threads_ex.dir/depend:
	cd /home/marc/workspace/dlib-master/examples/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/marc/workspace/dlib-master/examples /home/marc/workspace/dlib-master/examples /home/marc/workspace/dlib-master/examples/build /home/marc/workspace/dlib-master/examples/build /home/marc/workspace/dlib-master/examples/build/CMakeFiles/threads_ex.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/threads_ex.dir/depend

