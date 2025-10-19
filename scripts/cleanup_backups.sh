#!/bin/sh
#
# Cleanup helper for legacy backup/pilot files.
# Lists candidates, asks for confirmation, writes DELETED_BACKUPS.md
# and removes the files so they stay recoverable via git history.
#

set -e

echo "🔍 Scanning for backup / pilot files..."

FILES=$(
    find src -type f \( \
        -name '*backup*.py' -o \
        -name '*_pilot_b.py' -o \
        -name '*_pilot_a.py' -o \
        -name '*_old.py' -o \
        -name '*_deprecated.py' \
    \) ! -path '*/experiments/*'
)

if [ -z "$FILES" ]; then
    echo "✅ No backup or pilot files found."
    exit 0
fi

echo "The following files are candidates for deletion:"
nl <<EOF
$FILES
EOF

echo
printf "⚠️  Delete these files? (y/N): "
read ans

case "$ans" in
    y|Y)
        ;;
    *)
        echo "Cancelled."
        exit 0
        ;;
esac

if [ ! -f DELETED_BACKUPS.md ]; then
    cat > DELETED_BACKUPS.md <<'EOF'
# Deleted Backup / Pilot Files

The following files were removed by `scripts/cleanup_backups.sh`. Recover any file via git history:

```bash
# Example recovery
git log -- "path/to/file.py"
git checkout <commit> -- "path/to/file.py"
```

EOF
fi

printf "\n## %s\n" "$(date +%Y-%m-%d)" >> DELETED_BACKUPS.md
echo "$FILES" | while IFS= read -r file; do
    [ -n "$file" ] || continue
    printf "- %s\n" "$file" >> DELETED_BACKUPS.md
done

echo "$FILES" | while IFS= read -r file; do
    [ -n "$file" ] || continue
    rm -fv "$file"
done

echo "🗑️  Files deleted. Recovery instructions written to DELETED_BACKUPS.md"
echo "✅ Cleanup complete."
