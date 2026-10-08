; Inno Setup betiği — Windows kurulum dosyası üretir.
; Derleme: iscc /DAppVersion=1.0.0 installer\windows.iss   (önce PyInstaller çalışmış olmalı)

#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif

[Setup]
AppId={{6C2B7E1A-3F4D-4E8B-9A51-2D7C0F9E4B13}
AppName=Canlı Çevirmen (RO → TR)
AppVersion={#AppVersion}
AppPublisher=Kişisel kullanım
DefaultDirName={localappdata}\Programs\CanliCevirmen
DefaultGroupName=Canlı Çevirmen
; Yönetici izni istemez; yalnızca kuran kullanıcı için kurulur.
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist-installer
OutputBaseFilename=CanliCevirmen-Setup-Windows
; Modeller zaten sıkıştırılmış (int8); hızlı sıkıştırma yeterli.
Compression=lzma2/fast
SolidCompression=no
WizardStyle=modern
UninstallDisplayName=Canlı Çevirmen (RO → TR)
UninstallDisplayIcon={app}\CanliCevirmen.exe
DisableProgramGroupPage=yes

[Languages]
Name: "turkish"; MessagesFile: "compiler:Languages\Turkish.isl"

[Tasks]
Name: "desktopicon"; Description: "Masaüstüne kısayol oluştur"; GroupDescription: "Kısayollar:"

[Files]
Source: "..\dist\CanliCevirmen\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Canlı Çevirmen"; Filename: "{app}\CanliCevirmen.exe"
Name: "{group}\Kaldır"; Filename: "{uninstallexe}"
Name: "{userdesktop}\Canlı Çevirmen"; Filename: "{app}\CanliCevirmen.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\CanliCevirmen.exe"; Description: "Canlı Çevirmen'i başlat"; Flags: nowait postinstall skipifsilent
