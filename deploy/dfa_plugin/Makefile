# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.

# Compile src, build dir, target
SRC_PATH      :=	.
TARGET		  :=	$(ENVTARGETPLUGIN)
BUILD_PATH    :=	$(ENVBUILDDIR)

# Compile level flag
DEBUG		  :=	0
WARN  		  :=	0

# Compile src, build file, target
CXX_SRC       :=	$(wildcard $(SRC_PATH)/*.cpp)
CUDA_SRC   	  :=	$(wildcard $(SRC_PATH)/*.cu)
DEPS_INC	  :=	$(wildcard $(SRC_PATH)/*.h)

APP_DEPS      :=  	$(CXX_SRC)
APP_DEPS      +=  	$(CUDA_SRC)
APP_DEPS      +=  	$(DEPS_INC)

APP_OBJS      :=	$(patsubst $(SRC_PATH)%, $(BUILD_PATH)%, $(CXX_SRC:.cpp=.cpp.o))
APP_OBJS      +=	$(patsubst $(SRC_PATH)%, $(BUILD_PATH)%, $(CUDA_SRC:.cu=.cu.o))  
APP_MKS       :=  	$(APP_OBJS:.o=.mk)

# Compile options
CXXFLAGS      :=  	-std=c++17 -pthread -fPIC
CUDAFLAGS     :=  	--shared -Xcompiler -fPIC \
					-gencode=arch=compute_$(CUDASM),code=compute_$(CUDASM) \
					-gencode=arch=compute_$(CUDASM),code=sm_$(CUDASM)
INCS          :=  	-I $(SRC_PATH) \
					-I $(ENV_CUDA_INC) \
				  	-I $(ENV_TensorRT_INC)
LIBS          :=  	-L "$(ENV_CUDA_LIB)" \
                  	-L "$(ENV_TensorRT_LIB)" \
				  	-L "$(ENV_cuDNN_LIB)" \
                  	-lcudart \
					-lcublas \
				  	-lnvinfer \
					-lnvonnxparser \
				  	-lcudnn \
				  	-lstdc++fs

ifeq ($(DEBUG),1)
CUDAFLAGS     +=	-g -G
CXXFLAGS      +=  	-g -O0
else
CUDAFLAGS     +=  	-O3
CXXFLAGS      +=  	-O3
endif

ifeq ($(WARN),1)
CUDAFLAGS     +=  	-Wall -Wunused-function -Wunused-variable -Wfatal-errors
CXXFLAGS      +=  	-Wall -Wunused-function -Wunused-variable -Wfatal-errors
else
CUDAFLAGS     +=	-w
CXXFLAGS      +=  	-w
endif

# Compile
TARGET_PATH	  :=	$(firstword $(subst /, ,$(TARGET)))
all: 
	$(MAKE) $(TARGET)
$(TARGET): $(APP_DEPS) $(APP_OBJS)
	@mkdir -p $(TARGET_PATH)
	@nvcc $(CUDAFLAGS) $(APP_OBJS) -o $@ $(INCS) $(LIBS)
	@echo 5-Finish Compile Target : $(TARGET)!
$(BUILD_PATH)/%.cu.o: $(SRC_PATH)/%.cu
	@mkdir -p $(BUILD_PATH)
	@nvcc -o $@ -c $< $(CUDAFLAGS) $(INCS)
	@echo 4-Finish Compile CUDA Objects $@
$(BUILD_PATH)/%.cpp.o: $(SRC_PATH)/%.cpp 
	@mkdir -p $(BUILD_PATH)
	@g++ -o $@ -c $< $(CXXFLAGS) $(INCS)
	@echo 3-Finish Compile CXX Objects : $@
$(BUILD_PATH)/%.cpp.mk: $(SRC_PATH)/%.cpp
	@mkdir -p $(BUILD_PATH)
	@g++ -M $< -MF $@ -MT $(@:.cpp.mk=.cpp.o) $(CXXFLAGS) $(INCS) 
	@echo 2-Finish Compile CXX Make Policy $@
$(BUILD_PATH)/%.cu.mk: $(SRC_PATH)/%.cu
	@mkdir -p $(BUILD_PATH)
	@nvcc -M $< -MF $@ -MT $(@:.cu.mk=.cu.o) $(CUDAFLAGS)
	@echo 1-Finish Compile CUDA Make Policy $@

show: 
	@echo INCS		:	$(INCS)
	@echo LIBS		:	$(LIBS)
	@echo APP_DEPS	:	$(APP_DEPS)
	@echo APP_OBJS	:	$(APP_OBJS)
	@echo APP_MKS	:	$(APP_MKS)

clean:
	rm -r build
	rm -r lib

ifneq ($(MAKECMDGOALS), clean)
-include $(APP_MKS)
endif

.PHONY: all show clean 