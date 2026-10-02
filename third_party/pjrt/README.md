# third_party/pjrt

`pjrt_c_api.h`: PJRT C API のヘッダ（PJRT API 0.116）。

- 出所: openxla/xla の `xla/pjrt/c/pjrt_c_api.h`
  @ `81ee80d1899b665e4ae209b4c142e2b351a61d5e`
- sha256: `174b44736ccb200cb402c2466822220971ba578e1e8dfd690c56f822a9e71582`
- ライセンス: Apache License 2.0（ファイル冒頭の著作権表示のとおり。変更なし）
- 版の選び方: プラグインは自分の版以上の `struct_size` を受け付けるので、
  新しいヘッダ（0.116）は CPU プラグイン（0.81）でも CUDA プラグイン（0.115）
  でも使える。逆（古いヘッダで新しいプラグインの機能を使う）はできない。
  更新するときは、この README と docs/pjrt-setup.md を一緒に直す。
