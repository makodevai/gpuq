import gpuinfo as G


def main():
    for p in G.Provider:
        print(f'Has provider {p.name}:', G.hasprovider(p))
    print()

    all_ = G.query(visible_only=False)
    visible = G.query()

    print("All devices:")
    print("=====================")
    for gpu in all_:
        print(gpu)

    print()
    print("Visible devices:")
    print("=====================")
    for gpu in visible:
        print(gpu)


if __name__ == '__main__':
    main()
