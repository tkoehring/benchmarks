CC = gcc

SRC_DIR = src/yaksa
BUILD_DIR = build/yaksa
BIN_DIR = bin/yaksa

TARGET = pack_unpack

SRC_LIST = $(wildcard $(SRC_DIR)/*.c)
OBJ_LIST = $(BUILD_DIR)/$(notdir $(SRC_LIST:.c=.o))
 

INC_PATH = -I/home/tkoehring/Code/Yaksa/install/include
LIB_PATH = -L/home/tkoehring/Code/Yaksa/install/lib
LIB = -lyaksa

CFLAGS += $(INC_PATH)
LDFLAGS += $(LIB_PATH)

start: $(TARGET)

$(TARGET): $(OBJ_LIST)
	$(CC) $(CFLAGS) $^ -o $(BIN_DIR)/$@ $(LDFLAGS) $(LIB)

$(OBJ_LIST): $(SRC_LIST)
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: install
install:
	echo "installing.."

.PHONY: clean
clean:
	rm -f $(BIN_DIR)/$(TARGET) $(BUILD_DIR)/*.o
