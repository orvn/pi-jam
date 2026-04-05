VARIANTS = picojam jam jamxe jamkid

all: $(VARIANTS)

$(VARIANTS):
	$(MAKE) -C $@

clean:
	for v in $(VARIANTS); do $(MAKE) -C $$v clean; done

.PHONY: all clean $(VARIANTS)
