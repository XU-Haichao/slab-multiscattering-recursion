CXX ?= c++
HDF5_PREFIX ?= $(CONDA_PREFIX)
comma := ,
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pedantic
CPPFLAGS ?= $(if $(HDF5_PREFIX),-I$(HDF5_PREFIX)/include,)
LDFLAGS ?= $(if $(HDF5_PREFIX),-L$(HDF5_PREFIX)/lib -Wl$(comma)-rpath$(comma)$(HDF5_PREFIX)/lib,)
LDLIBS ?= -lhdf5_cpp -lhdf5

SOURCES = main.cpp config.cpp electron.cpp histogram.cpp kinematics.cpp kn_scatter.cpp \
          photon.cpp slab.cpp slab_injection.cpp slab_transport.cpp thermal_electron.cpp \
          thermal_kn_transport.cpp vec3.cpp
TARGET = mcrt_ang

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SOURCES) $(LDFLAGS) $(LDLIBS) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean
