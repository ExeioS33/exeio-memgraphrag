import { describe, expect, it } from 'vitest'

import { bucketFor, groupThreads, matchesFilter } from './threads'

const NOW = 1_800_000_000
const H = 3600
const D = 24 * H

const thread = (title: string, ago: number) => ({ title, updated_at: NOW - ago })

describe('matchesFilter', () => {
  it('matches case- and accent-insensitively', () => {
    expect(matchesFilter({ title: 'Budget prévisionnel 2026' }, 'PREVISION')).toBe(true)
    expect(matchesFilter({ title: 'Budget prévisionnel 2026' }, 'ré')).toBe(true)
  })

  it('matches everything on an empty or blank filter', () => {
    expect(matchesFilter({ title: 'x' }, '')).toBe(true)
    expect(matchesFilter({ title: 'x' }, '   ')).toBe(true)
  })

  it('rejects a title without the needle', () => {
    expect(matchesFilter({ title: 'Contrats' }, 'budget')).toBe(false)
  })
})

describe('bucketFor', () => {
  it('names the sidebar buckets by age', () => {
    expect(bucketFor(NOW - 2 * H, NOW)).toBe("Aujourd'hui")
    expect(bucketFor(NOW - 30 * H, NOW)).toBe('Hier')
    expect(bucketFor(NOW - 3 * D, NOW)).toBe('7 derniers jours')
    expect(bucketFor(NOW - 20 * D, NOW)).toBe('30 derniers jours')
    expect(bucketFor(NOW - 90 * D, NOW)).toBe('Plus ancien')
  })
})

describe('groupThreads', () => {
  const threads = [
    thread('Budget', 1 * H),
    thread('Contrats', 30 * H),
    thread('Budget 2025', 40 * D),
    thread('Fournisseurs', 3 * D),
  ]

  it('keeps bucket order and drops empty buckets', () => {
    const groups = groupThreads(threads, '', NOW)
    expect(groups.map(([bucket]) => bucket)).toEqual([
      "Aujourd'hui",
      'Hier',
      '7 derniers jours',
      'Plus ancien',
    ])
  })

  it('filters before grouping, so a bucket with no match disappears', () => {
    const groups = groupThreads(threads, 'budget', NOW)
    expect(groups.map(([bucket, list]) => [bucket, list.map((t) => t.title)])).toEqual([
      ["Aujourd'hui", ['Budget']],
      ['Plus ancien', ['Budget 2025']],
    ])
  })

  it('returns nothing when nothing matches', () => {
    expect(groupThreads(threads, 'zzz', NOW)).toEqual([])
  })
})
