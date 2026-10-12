//! Export of the finished rows: CSV (one line per symbol, as the table)
//! and JSON (the CLI's `--json` array).

use std::path::Path;

use dm_reader::{ReadResult, printable, to_json};

use crate::table::{Row, Status, grade};

/// Excel needs the BOM to read UTF-8 (non-ASCII file names).
const BOM: &str = "\u{feff}";
const HEADER: &str = "file,symbol,decoded,text,engine,size,grade,time_s,note";

/// First characters that make a spreadsheet read a cell as a formula
/// (OWASP CSV injection); such cells get a leading `'`.
const FORMULA_START: [char; 6] = ['=', '+', '-', '@', '\t', '\r'];

fn field(s: &str) -> String {
    let s = if s.starts_with(FORMULA_START) {
        format!("'{s}")
    } else {
        s.to_string()
    };
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s
    }
}

fn line(cols: &[String]) -> String {
    cols.iter().map(|c| field(c)).collect::<Vec<_>>().join(",")
}

fn size(r: &ReadResult) -> String {
    r.fit.as_ref().map(|f| f.geom.name()).unwrap_or_default()
}

/// CSV of the finished rows (queued and running ones are left out).
pub fn csv(rows: &[Row]) -> String {
    let mut out = vec![format!("{BOM}{HEADER}")];
    for row in rows {
        let file = row.path.display().to_string();
        match &row.status {
            Status::Done(r) if !r.data.is_empty() => {
                for (k, s) in std::iter::once(&**r).chain(&r.more).enumerate() {
                    out.push(line(&[
                        file.clone(),
                        (k + 1).to_string(),
                        "yes".into(),
                        printable(&s.data, s.text_dec()),
                        s.engine.clone(),
                        size(s),
                        grade(s),
                        // the whole image's time on its first symbol
                        if k == 0 {
                            format!("{:.2}", r.elapsed)
                        } else {
                            String::new()
                        },
                        String::new(),
                    ]));
                }
            }
            Status::Done(r) => out.push(line(&[
                file,
                String::new(),
                "no".into(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                format!("{:.2}", r.elapsed),
                r.near_miss
                    .as_ref()
                    .map(|n| format!("near miss {}", n.nm.size))
                    .unwrap_or_default(),
            ])),
            Status::Error(e) => out.push(line(&[
                file,
                String::new(),
                "no".into(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                e.clone(),
            ])),
            Status::Stopped => out.push(line(&[
                file,
                String::new(),
                "no".into(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                "stopped".into(),
            ])),
            Status::Queued | Status::Running(..) => {}
        }
    }
    out.join("\r\n") + "\r\n"
}

/// JSON of the read images, as `dpmread --json` (load errors and stopped
/// images are not in it, as in the CLI).
pub fn json(rows: &[Row]) -> String {
    let results: Vec<ReadResult> = rows
        .iter()
        .filter_map(|r| match &r.status {
            Status::Done(res) => Some((**res).clone()),
            _ => None,
        })
        .collect();
    to_json(&results) + "\n"
}

/// Write CSV or JSON by the file extension (`.json`, else CSV).
pub fn write(path: &Path, rows: &[Row]) -> std::io::Result<()> {
    let is_json = path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("json"));
    let text = if is_json { json(rows) } else { csv(rows) };
    std::fs::write(path, text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn done(id: u64, name: &str, data: &[u8], more: &[&[u8]]) -> Row {
        let one = |d: &[u8]| ReadResult {
            path: PathBuf::from(name),
            engine: "zxing".into(),
            data: d.to_vec(),
            text: String::from_utf8_lossy(d).into_owned(),
            elapsed: 1.5,
            ..Default::default()
        };
        let mut r = one(data);
        r.more = more.iter().map(|d| one(d)).collect();
        Row {
            id,
            path: PathBuf::from(name),
            status: Status::Done(Box::new(r)),
        }
    }

    fn rows() -> Vec<Row> {
        vec![
            done(0, "/x/Kuva ÄÖ 1.png", b"A,\"B\"", &[b"C"]),
            done(1, "/x/b.png", b"", &[]),
            Row {
                id: 2,
                path: "/x/c.png".into(),
                status: Status::Error("cannot read".into()),
            },
            Row {
                id: 3,
                path: "/x/d.png".into(),
                status: Status::Stopped,
            },
            Row {
                id: 4,
                path: "/x/e.png".into(),
                status: Status::Queued,
            },
        ]
    }

    #[test]
    fn csv_lines_per_symbol() {
        let c = csv(&rows());
        let lines: Vec<&str> = c.split("\r\n").collect();
        assert_eq!(lines[0], format!("{BOM}{HEADER}"));
        assert_eq!(
            lines[1],
            "/x/Kuva ÄÖ 1.png,1,yes,\"A,\"\"B\"\"\",zxing,,,1.50,"
        );
        assert_eq!(lines[2], "/x/Kuva ÄÖ 1.png,2,yes,C,zxing,,,,");
        assert_eq!(lines[3], "/x/b.png,,no,,,,,1.50,");
        assert_eq!(lines[4], "/x/c.png,,no,,,,,,cannot read");
        assert_eq!(lines[5], "/x/d.png,,no,,,,,,stopped");
        assert_eq!(lines[6], "");
        assert_eq!(lines.len(), 7);
    }

    #[test]
    fn csv_escapes_formulas() {
        let rows = vec![
            done(0, "=cmd.png", b"=1+2", &[b"+A", b"-B", b"@C", b"x=y"]),
            done(1, "@ä.png", b"", &[]),
        ];
        let c = csv(&rows);
        let lines: Vec<&str> = c.split("\r\n").collect();
        assert_eq!(lines[1], "'=cmd.png,1,yes,'=1+2,zxing,,,1.50,");
        assert_eq!(lines[2], "'=cmd.png,2,yes,'+A,zxing,,,,");
        assert_eq!(lines[3], "'=cmd.png,3,yes,'-B,zxing,,,,");
        assert_eq!(lines[4], "'=cmd.png,4,yes,'@C,zxing,,,,");
        assert_eq!(lines[5], "'=cmd.png,5,yes,x=y,zxing,,,,");
        assert_eq!(lines[6], "'@ä.png,,no,,,,,1.50,");
        assert_eq!(field("\tD"), "'\tD");
        assert_eq!(field("\rD"), "\"'\rD\"");
        assert_eq!(field("-1,2"), "\"'-1,2\"");
    }

    #[test]
    fn json_is_the_cli_array() {
        let rows = rows();
        let v: serde_json::Value = serde_json::from_str(&json(&rows)).unwrap();
        let a = v.as_array().unwrap();
        assert_eq!(a.len(), 2);
        let done: Vec<ReadResult> = rows[..2]
            .iter()
            .map(|r| match &r.status {
                Status::Done(x) => (**x).clone(),
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(json(&rows), to_json(&done) + "\n");
    }

    #[test]
    fn write_by_extension() {
        let d = std::env::temp_dir().join(format!("dm-gui-export-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let rows = rows();
        write(&d.join("r.JSON"), &rows).unwrap();
        write(&d.join("r.csv"), &rows).unwrap();
        assert!(
            std::fs::read_to_string(d.join("r.JSON"))
                .unwrap()
                .starts_with('[')
        );
        assert!(
            std::fs::read_to_string(d.join("r.csv"))
                .unwrap()
                .starts_with(BOM)
        );
        let _ = std::fs::remove_dir_all(&d);
    }
}
