{ pkgs }: {
  deps = [
    pkgs.ffmpeg
    pkgs.python311
    pkgs.python311Packages.pip
  ];
}
