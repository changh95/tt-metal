/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace tt::tt_metal::detail
{
    struct HashLookup {
    static HashLookup& inst() {
        static HashLookup inst_;
        return inst_;
    }

    bool exists(size_t khash) {
        unique_lock<mutex> lock(mutex_);
        return hashes_.find(khash) != hashes_.end();
    }
    bool add(size_t khash) {
        unique_lock<mutex> lock(mutex_);
        bool ret = false;
        if (hashes_.find(khash) == hashes_.end() ){
            hashes_.insert(khash);
            ret = true;
        }
        return ret;
    }

    void clear() {
        unique_lock<mutex> lock(mutex_);
        hashes_.clear();
    }


    private:
        std::mutex mutex_;
        std::unordered_set<size_t > hashes_;
    };


    /**
     * Clear the current kernel compilation cache.
     *
     * Return value: void
     */
    inline void ClearKernelCache()
    {
        detail::HashLookup::inst().clear();
    }
}
