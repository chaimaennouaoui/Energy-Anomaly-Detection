import os, shutil, time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARCHIVE_DIR  = os.path.join(PROJECT_ROOT, "streaming", "archive")
INCOMING_DIR = os.path.join(PROJECT_ROOT, "streaming", "incoming")

DELAY_SECONDS = 5
MAX_FILES = 50

files = sorted([f for f in os.listdir(ARCHIVE_DIR) if f.endswith(".csv")])

if not files:
    print("❌ Aucun fichier dans archive. Lance generate_stream_batches.py d'abord.")
    exit()

print(f"✅ {len(files)} fichiers trouvés dans archive.")
print(f"✅ Envoi d'un fichier toutes les {DELAY_SECONDS} secondes...\n")

count = 0
for f in files:
    if count >= MAX_FILES:
        break

    shutil.copy(os.path.join(ARCHIVE_DIR, f), os.path.join(INCOMING_DIR, f))
    count += 1
    print(f"✅ envoyé {f} ({count}/{MAX_FILES})")
    time.sleep(DELAY_SECONDS)

print("\n🎉 Terminé !")
