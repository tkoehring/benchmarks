CC = gcc

SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin

YAKSA_SRC_LIST := $(wildcard $(SRC_DIR)/yaksa/*.c)
YAKSA_OBJ_LIST := $(patsubst $(SRC_DIR)/yaksa/%.c, $(BUILD_DIR)/yaksa/%.o, $(YAKSA_SRC_LIST))
#$(BUILD_DIR)/yaksa/$(notdir $(YAKSA_SRC_LIST:%.c=%.o))

YAKSA_INSTALL_PATH = $(HOME)/Code/Yaksa/install
YAKSA_INC_PATH = -I$(YAKSA_INSTALL_PATH)/include
YAKSA_LIB_PATH = -L$(YAKSA_INSTALL_PATH)/lib

YAKSA_CFLAGS =
YAKSA_LDFLAGS += $(YAKSA_LIB_PATH)
YAKSA_LIB = -lyaksa

yaksa: $(YAKSA_OBJ_LIST)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@echo $<
	@echo $@
	@mkdir -p $(BUILD_DIR)/yaksa
	$(CC) $(YAKSA_CFLAGS) $(YAKSA_INC_PATH) -c $< -o $@

.PHONY: install_yaksa
install_yaksa: $(YAKSA_OBJ_LIST)
	@mkdir -p $(BIN_DIR)/yaksa
	$(CC) $^ -o $(BIN_DIR)/yaksa/$(TARGET) $(YAKSA_LDFLAGS) $(YAKSA_LIB)

.PHONY: clean
clean:
	rm -rf $(BIN_DIR) $(BUILD_DIR)

test:
	@echo $(YAKSA_SRC_LIST)
	@echo $(YAKSA_OBJ_LIST)
