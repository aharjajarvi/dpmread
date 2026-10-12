//! Input list of the GUI: dropped or picked files and folders, expanded with
//! the CLI's `collect_inputs` and with duplicates ignored.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use dm_reader::{IMAGE_EXT, collect_inputs};

/// Extensions for the file dialog filter (`IMAGE_EXT` without the dot).
pub fn dialog_extensions() -> Vec<&'static str> {
    IMAGE_EXT
        .iter()
        .map(|e| e.trim_start_matches('.'))
        .collect()
}

/// Identity of a path for the duplicate check: the canonical path when it
/// exists, so `a/../a/x.png` and a symlink count as the same file.
fn key(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Every image added so far (only their identities: the rows keep the order).
#[derive(Default)]
pub struct Inputs {
    seen: HashSet<PathBuf>,
}

impl Inputs {
    /// Add files and folders (folders expand to their image files, sorted
    /// by name); paths already in the list are skipped. Returns the paths
    /// that were new, in order.
    pub fn add(&mut self, dropped: &[PathBuf]) -> Vec<PathBuf> {
        collect_inputs(dropped)
            .into_iter()
            .filter(|p| self.seen.insert(key(p)))
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }

    pub fn clear(&mut self) {
        self.seen.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("dm-gui-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn folder_expands_and_duplicates_are_ignored() {
        let d = tmp_dir("inputs");
        for n in ["b.png", "Kuva ÄÖ 1.PNG", "a.jpg", "notes.txt"] {
            std::fs::write(d.join(n), b"x").unwrap();
        }
        let mut inp = Inputs::default();
        let new = inp.add(std::slice::from_ref(&d));
        let names: Vec<_> = new
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["Kuva ÄÖ 1.PNG", "a.jpg", "b.png"]);
        // the same folder, a file in it by another spelling, and a new file
        std::fs::write(d.join("c.bmp"), b"x").unwrap();
        let again = inp.add(&[d.clone(), d.join(".").join("a.jpg")]);
        assert_eq!(again, [d.join("c.bmp")]);
        inp.clear();
        assert!(inp.is_empty());
        assert_eq!(inp.add(&[d.join("b.png")]).len(), 1);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn missing_file_is_kept_once() {
        // an unreadable file shows up as a failed row later, not silently dropped
        let mut inp = Inputs::default();
        let p = PathBuf::from("/nonexistent/x.png");
        assert_eq!(inp.add(&[p.clone(), p.clone()]), [p]);
    }

    #[test]
    fn extensions_have_no_dot() {
        assert!(dialog_extensions().contains(&"png"));
        assert!(dialog_extensions().iter().all(|e| !e.starts_with('.')));
    }
}
