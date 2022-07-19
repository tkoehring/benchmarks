CC=gcc
NVCC=nvcc

SRC_DIR=src
BUILD_DIR=build
BIN_DIR=bin

GNU_SRC_LIST=$(wildcard $(SRC_DIR)/gnu/*.c)
GNU_OBJ_LIST=$(patsubst %.c, $(BUILD_DIR)/gnu/%.o, $(notdir $(GNU_SRC_LIST)))
GNU_TARGET_LIST=$(patsubst %.o, $(BIN_DIR)/gnu/%, $(notdir $(GNU_OBJ_LIST)))

CUDA_SRC_LIST=$(wildcard $(SRC_DIR)/cuda/*.c)
CUDA_OBJ_LIST=$(patsubst %.c, $(BUILD_DIR)/cuda/%.o, $(notdir $(CUDA_SRC_LIST)))
CUDA_TARGET_LIST=$(patsubst %.o, $(BIN_DIR)/cuda/%, $(notdir $(CUDA_OBJ_LIST)))

YAKSA_SRC_LIST=$(wildcard $(SRC_DIR)/yaksa/*.c)
YAKSA_OBJ_LIST=$(patsubst %.c, $(BUILD_DIR)/yaksa/%.o, $(notdir $(YAKSA_SRC_LIST)))
YAKSA_TARGET_LIST=$(patsubst %.o, $(BIN_DIR)/yaksa/%, $(notdir $(YAKSA_OBJ_LIST)))

## CHANGE ME
## These variables need to be manually changed to match your yaksa installation
## location
YAKSA_INSTALL_PATH=$(HOME)/Code/Yaksa/install
YAKSA_INC_PATH=-I$(YAKSA_INSTALL_PATH)/include
YAKSA_LIB_PATH=-L$(YAKSA_INSTALL_PATH)/lib

## CHANGE ME
## These variables need to be manually changed to match your cuda installation
## location
CUDA_INSTALL_PATH=/usr/local/cuda
CUDA_INC_PATH=-I$(CUDA_INSTALL_PATH)/include
CUDA_LIB_PATH=-L$(CUDA_INSTALL_PATH)/lib

CFLAGS=
INC_PATH += $(YAKSA_INC_PATH)
INC_PATH += $(CUDA_INC_PATH)
LIB_PATH += $(YAKSA_LIB_PATH)
LIB_PATH += $(CUDA_LIB_PATH)
LDFLAGS += $(YAKSA_LIB_PATH)
LDFLAGS += $(CUDA_LIB_PATH)
LIB=-lyaksa

## MAKE EVERYTHING
.PHONY: all
all: gnu cuda yaksa

## GNU PROGRAMS
.PHONY: gnu
gnu: $(GNU_OBJ_LIST)

$(BUILD_DIR)/gnu/%.o: $(SRC_DIR)/gnu/%.c
	@mkdir -p $(BUILD_DIR)/gnu
	$(CC) $(CFLAGS) -c $< -o $@

## CUDA PROGRAMS
.PHONY: cuda
cuda: $(CUDA_OBJ_LIST)

$(BUILD_DIR)/cuda/%.o: $(SRC_DIR)/cuda/%.c
	@mkdir -p $(BUILD_DIR)/cuda
	$(NVCC) $(CFLAGS) $(CUDA_INC_PATH) -c $< -o $@

## YAKSA PROGRAMS
.PHONY: yaksa
yaksa: $(YAKSA_OBJ_LIST)

$(BUILD_DIR)/yaksa/%.o: $(SRC_DIR)/yaksa/%.c
	@mkdir -p $(BUILD_DIR)/yaksa
	$(NVCC) $(CFLAGS) $(INC_PATH) -c $< -o $@

## INSTALL
.PHONY: install
install: $(GNU_TARGET_LIST) $(CUDA_TARGET_LIST) $(YAKSA_TARGET_LIST) 

$(BIN_DIR)/gnu/%: $(BUILD_DIR)/gnu/%.o
	@mkdir -p $(BIN_DIR)/gnu
	$(CC) $< -o $@ -lm

$(BIN_DIR)/cuda/%: $(BUILD_DIR)/cuda/%.o
	@mkdir -p $(BIN_DIR)/cuda
	$(NVCC) $< -o $@ $(CUDA_LIB_PATH)

$(BIN_DIR)/yaksa/%: $(BUILD_DIR)/yaksa/%.o
	@mkdir -p $(BIN_DIR)/yaksa
	$(NVCC) $< -o $@ $(LDFLAGS) $(LIB)



.PHONY: clean
clean:
	rm -rf $(BIN_DIR) $(BUILD_DIR)
