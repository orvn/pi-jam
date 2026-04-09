CC      ?= gcc
CFLAGS  = -O2 -Wall
TARGET  = dist/neuralpi
DATA    = td_transformer.txt

$(TARGET): inference.c theme.h | dist
	$(CC) $(CFLAGS) -o $@ inference.c -lm

theme.h: $(DATA)
	../gen_theme.sh $(DATA) theme.h

dist:
	mkdir -p dist

clean:
	rm -f $(TARGET) theme.h *.o

.PHONY: clean
