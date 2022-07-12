CC = gcc

SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin

GNU_SRC_LIST = $(wildcard $(SRC_DIR)/gnu/*.c)
GNU_OBJ_LIST = $(patsubst %.c, $(BUILD_DIR)/gnu/%.o, $(notdir $(GNU_SRC_LIST)))
GNU_TARGET_LIST = $(patsubst %.o, $(BIN_DIR)/gnu/%, $(notdir $(GNU_OBJ_LIST)))

YAKSA_SRC_LIST = $(wildcard $(SRC_DIR)/yaksa/*.c)
YAKSA_OBJ_LIST = $(patsubst %.c, $(BUILD_DIR)/yaksa/%.o, $(notdir $(YAKSA_SRC_LIST)))
YAKSA_TARGET_LIST = $(patsubst %.o, $(BIN_DIR)/yaksa/%, $(notdir $(YAKSA_OBJ_LIST)))

YAKSA_INSTALL_PATH = $(HOME)/Code/Yaksa/install
YAKSA_INC_PATH = -I$(YAKSA_INSTALL_PATH)/include
YAKSA_LIB_PATH = -L$(YAKSA_INSTALL_PATH)/lib

YAKSA_CFLAGS =
YAKSA_LDFLAGS += $(YAKSA_LIB_PATH)
YAKSA_LIB = -lyaksa

## MAKE EVERYTHING
.PHONY: all
all: gnu yaksa

## GNU PROGRAMS
.PHONY: gnu
gnu: $(GNU_OBJ_LIST)


$(BUILD_DIR)/gnu/%.o: $(SRC_DIR)/gnu/%.c
	@mkdir -p $(BUILD_DIR)/gnu
	$(CC) -c $< -o $@

## YAKSA PROGRAMS
.PHONY: yaksa
yaksa: $(YAKSA_OBJ_LIST)

$(BUILD_DIR)/yaksa/%.o: $(SRC_DIR)/yaksa/%.c
	@mkdir -p $(BUILD_DIR)/yaksa
	$(CC) $(YAKSA_CFLAGS) $(YAKSA_INC_PATH) -c $< -o $@

## INSTALL
.PHONY: install
install: $(YAKSA_TARGET_LIST) $(GNU_TARGET_LIST)

$(BIN_DIR)/yaksa/%: $(BUILD_DIR)/yaksa/%.o
	@mkdir -p $(BIN_DIR)/yaksa
	$(CC) $< -o $@ $(YAKSA_LDFLAGS) $(YAKSA_LIB)

$(BIN_DIR)/gnu/%: $(BUILD_DIR)/gnu/%.o
	@mkdir -p $(BIN_DIR)/gnu
	$(CC) $< -o $@

.PHONY: clean
clean:
	rm -rf $(BIN_DIR) $(BUILD_DIR)
