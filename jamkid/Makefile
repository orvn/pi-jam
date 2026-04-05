CC      = gcc
CFLAGS  = -O2 -Wall
TARGET  = dist/jamkid
DATA    = td_jamkid.txt
SRC     = ../jamxe/inference.c

$(TARGET): $(SRC) theme.h | dist
	$(CC) $(CFLAGS) -include $(abspath theme.h) -DWEIGHTS_FILE='"weights_jamkid.bin"' -o $@ $(SRC)

theme.h: $(DATA)
	../gen_theme.sh $(DATA) theme.h

dist:
	mkdir -p dist

clean:
	rm -f $(TARGET) theme.h *.o

.PHONY: clean
