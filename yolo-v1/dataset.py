class YOLOv1Dataset(Dataset):
    def __init__(self, img_dir, lbl_dir, S=7, B=2, C=3, transform=None):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.S, self.B, self.C = S, B, C
        self.transform = transform
        self.files = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # ── image ──────────────────────────────────────────────
        img_path = os.path.join(self.img_dir, self.files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)   # → (3, 448, 448)

        # ── target tensor (S, S, 5*B + C) ──────────────────────
        S, B, C = self.S, self.B, self.C
        target = torch.zeros(S, S, 5 * B + C)

        lbl_path = os.path.join(
            self.lbl_dir,
            self.files[idx].replace(".jpg", ".txt")
        )
        with open(lbl_path) as f:
            for line in f:
                cls_id, xc, yc, w, h = map(float, line.split())
                col = int(xc * S)          # grid cell column
                row = int(yc * S)          # grid cell row
                x_cell = xc * S - col      # x offset inside cell
                y_cell = yc * S - row      # y offset inside cell

                if target[row, col, 4] == 0:   # cell not yet taken
                    box = torch.tensor([x_cell, y_cell, w, h, 1.0])
                    target[row, col, 0:5] = box     # box 1
                    target[row, col, 5:10] = box    # box 2 (same GT)
                    target[row, col, 10 + int(cls_id)] = 1.0  # one-hot

        return image, target   # (3,448,448)  (7,7,13)
