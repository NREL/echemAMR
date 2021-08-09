USE_EB = TRUE
DEBUG = FALSE
USE_MPI  = TRUE
USE_OMP  = FALSE

USE_HYPRE = FALSE

COMP = gnu

DIM = 3

AMREX_HOME ?= ../../../amrex

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package
include ../../src/Make.package

INCLUDE_LOCATIONS += ../../src
VPATH_LOCATIONS += ../../src

Pdirs := Base Boundary AmrCore
Pdirs += EB
Pdirs += LinearSolvers/MLMG

Ppack	+= $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir)/Make.package)

include $(Ppack)

include $(AMREX_HOME)/Tools/GNUMake/Make.rules

