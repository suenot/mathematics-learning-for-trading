/**
 * sync-content.mjs
 * Fetches chapter content from GitHub repos at build time.
 * Each chapter is stored in its own repo: NNN-slug
 * Supported file names: chapter.en.md / README.en.md / en.md (and .ru.md variants)
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DOCS_DIR = path.resolve(__dirname, 'src/content/docs');
const GH_RAW = 'https://raw.githubusercontent.com/suenot';

const CHAPTERS = [
  { num: '001', slug: '001-stochastic-calculus', dir: '01-stochastic-calculus' },
  { num: '002', slug: '002-market-microstructure', dir: '02-market-microstructure' },
  { num: '003', slug: '003-portfolio-optimization', dir: '03-portfolio-optimization' },
  { num: '004', slug: '004-ml-time-series', dir: '04-ml-time-series' },
  { num: '005', slug: '005-low-latency-systems', dir: '05-low-latency-systems' },
];

// Candidate file names to try, in order of preference
const EN_CANDIDATES = ['chapter.en.md', 'README.en.md', 'en.md', 'README.md'];
const RU_CANDIDATES = ['chapter.ru.md', 'README.ru.md', 'ru.md'];

function getTitle(content) {
  const h1 = content.match(/^#\s+(.+)$/m);
  return h1 ? h1[1].trim() : 'Chapter';
}

function stripFrontmatter(content) {
  if (content.startsWith('---')) {
    const end = content.indexOf('---', 3);
    if (end !== -1) return content.slice(end + 3).trimStart();
  }
  return content;
}

function rewriteImageUrls(content, repoSlug) {
  const base = `${GH_RAW}/${repoSlug}/main`;
  return content.replace(
    /!\[([^\]]*)\]\((?!https?:\/\/)([^)]+)\)/g,
    (_, alt, src) => `![${alt}](${base}/${src.replace(/^\.\//, '')})`
  );
}

function stripH1(content) {
  return content.replace(/^#\s+.+\n+/, '');
}

function addFrontmatter(title, content, repoSlug) {
  const body = stripH1(rewriteImageUrls(stripFrontmatter(content), repoSlug));
  return `---\ntitle: "${title.replace(/"/g, '\\"')}"\n---\n\n${body}`;
}

async function fetchText(url) {
  const res = await fetch(url);
  if (!res.ok) return null;
  return res.text();
}

async function fetchFirstAvailable(repoSlug, candidates) {
  for (const filename of candidates) {
    const url = `${GH_RAW}/${repoSlug}/main/${filename}`;
    const text = await fetchText(url);
    if (text) return text;
  }
  return null;
}

console.log('Syncing chapter content from GitHub...');

let written = 0;
let failed = 0;

for (const chapter of CHAPTERS) {
  const [enContent, ruContent] = await Promise.all([
    fetchFirstAvailable(chapter.slug, EN_CANDIDATES),
    fetchFirstAvailable(chapter.slug, RU_CANDIDATES),
  ]);

  for (const [lang, content] of [['en', enContent], ['ru', ruContent]]) {
    if (!content) {
      console.warn(`  ⚠ No ${lang} content for ${chapter.slug}`);
      failed++;
      continue;
    }

    const title = getTitle(content);
    const final = addFrontmatter(title, content, chapter.slug);
    const destDir = path.join(
      DOCS_DIR,
      lang === 'en' ? '' : lang,
      'chapters',
      chapter.dir
    );
    fs.mkdirSync(destDir, { recursive: true });
    fs.writeFileSync(path.join(destDir, `${chapter.slug}.md`), final, 'utf-8');
    written++;
    console.log(`  ✓ ${lang}: ${chapter.slug}`);
  }
}

console.log(`\nSynced ${written} files (${failed} missing)`);
