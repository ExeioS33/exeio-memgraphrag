import type { ChatThread } from '../api/types'

const DAY = 86_400

export const BUCKETS = ["Aujourd'hui", 'Hier', '7 derniers jours', '30 derniers jours', 'Plus ancien'] as const
export type Bucket = (typeof BUCKETS)[number]

export function bucketFor(updatedAt: number, now: number): Bucket {
  const age = now - updatedAt
  if (age < DAY) return "Aujourd'hui"
  if (age < 2 * DAY) return 'Hier'
  if (age < 7 * DAY) return '7 derniers jours'
  if (age < 30 * DAY) return '30 derniers jours'
  return 'Plus ancien'
}

/** Case- and accent-insensitive substring match on the title. Threads are all
 *  loaded for the sidebar anyway, so this needs no server round-trip. */
export function matchesFilter(thread: Pick<ChatThread, 'title'>, filter: string): boolean {
  const needle = fold(filter)
  return !needle || fold(thread.title).includes(needle)
}

function fold(text: string): string {
  return text
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .trim()
}

/** Filter, then group by age, keeping the sidebar's bucket order and dropping
 *  empty buckets. Threads inside a bucket keep the order they were given. */
export function groupThreads<T extends Pick<ChatThread, 'title' | 'updated_at'>>(
  threads: T[],
  filter: string,
  now = Math.floor(Date.now() / 1000),
): Array<[Bucket, T[]]> {
  const buckets = new Map<Bucket, T[]>()
  for (const thread of threads) {
    if (!matchesFilter(thread, filter)) continue
    const key = bucketFor(thread.updated_at, now)
    const list = buckets.get(key)
    if (list) list.push(thread)
    else buckets.set(key, [thread])
  }
  return BUCKETS.filter((k) => buckets.has(k)).map((k) => [k, buckets.get(k)!])
}
