; JAMXE 512-input feature layout
; This file is the canonical runtime-side reference for the target XE model.
; It is not yet wired into the stable runtime, but future jamxe runtime work
; should use these block boundaries instead of re-inventing them ad hoc.

N_IN_TARGET = 512
N_OUT_TARGET = 45
N_HID_TARGET = 512

; Live context
XE_LIVE_TRI_BASE = 0
XE_LIVE_TRI_SIZE = 192
XE_LIVE_BAG_BASE = 192
XE_LIVE_BAG_SIZE = 32
XE_LIVE_SFX_BASE = 224
XE_LIVE_SFX_SIZE = 32

; Frozen query anchor
XE_QTRI_BASE = 256
XE_QTRI_SIZE = 128
XE_QBAG_BASE = 384
XE_QBAG_SIZE = 32

; Answer/query side helpers
XE_ABAG_BASE = 416
XE_ABAG_SIZE = 32
XE_QSFX_BASE = 448
XE_QSFX_SIZE = 32
XE_QACT_BASE = 480
XE_QACT_SIZE = 32

; Expected packed weight sizes for the target XE model
XE_W1_INT2_SIZE = 65536
XE_W2_INT4_SIZE = 11520
XE_B1_SIZE = 512
XE_B2_SIZE = 45
XE_B2S_SIZE = 45
XE_B2M_SIZE = 45
XE_BLOB_SIZE = 77703
